import warnings

import numpy as np
from sklearn.utils.extmath import randomized_svd

from djimaging.utils.image_utils import resize_image


def compute_explained_rf(rf, rf_fit):
    return 1. - (np.var(rf - rf_fit) / np.var(rf))


def smooth_rf(rf, blur_std, blur_npix):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(rf, mode='nearest', sigma=blur_std, truncate=np.floor(blur_npix / blur_std), order=0)


def resize_srf(srf, scale=None, output_shape=None):
    """Change size of sRF, used for up or down sampling"""
    if scale is not None:
        output_shape = np.array(srf.shape) * scale
    else:
        assert output_shape is not None
    return resize_image(srf, output_shape=output_shape, order=1)


def merge_strf(srf, trf):
    """Reconstruct STRF from sRF and tRF"""
    assert trf.ndim == 1, trf.ndim
    assert srf.ndim == 2, srf.ndim
    rf = np.kron(trf, srf.flat).reshape(trf.shape + srf.shape)
    return rf


def split_strf(strf, method='SVD', blur_std: float = 0, blur_npix: int = 1, upsample_srf_scale: int = 0):
    """Split STRF into sRF and tRF"""
    if blur_std > 0:
        strf = np.stack([smooth_rf(rf=srf_i, blur_std=blur_std, blur_npix=blur_npix) for srf_i in strf])

    explained_variance = None
    if method.lower() == 'svd':
        srf, trf, explained_variance = split_rf_svd(strf)
    elif method.lower() == 'sd':
        srf, trf = split_rf_sd(strf)
    elif method.lower() == 'peak':
        srf, trf = split_rf_peak(strf)
    else:
        raise NotImplementedError(f"Method {method} for RF split not implemented")

    if upsample_srf_scale > 1:
        srf = resize_srf(srf, scale=upsample_srf_scale)

    return srf, trf, explained_variance


def split_rf_svd(strf):
    """
    Assuming an RF is time-space separable, get spatial and temporal filters using SVD.

    This function was modified from RFEst, which is licensed under
    the GNU General Public License version 3.0 (GNU GPL v3.0).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

    For details see: https://github.com/berenslab/RFEst

    """
    dims = strf.shape

    if len(dims) == 3:
        dims_t_rf = dims[0]
        dims_s_rf = dims[1:]
        u, s, vt = randomized_svd(strf.reshape(dims_t_rf, -1), 3, random_state=0)
        srf = vt[0].reshape(*dims_s_rf)
        trf = u[:, 0]

        explained_variance = s[0] ** 2 / np.sum(s ** 2)

        srf, trf = rescale_trf_srf(srf, trf, strf)

    elif len(dims) == 2:
        dims_t_rf = dims[0]
        dims_s_rf = dims[1]
        u, s, vt = randomized_svd(strf.reshape(dims_t_rf, dims_s_rf), 3, random_state=0)
        srf = vt[0]
        trf = u[:, 0]

        explained_variance = s[0] ** 2 / np.sum(s ** 2)

        srf, trf = rescale_trf_srf(srf, trf, strf)

    elif len(dims) == 4:
        srfs, trfs, explained_variances = [], [], []
        for i in range(dims[-1]):
            srf, trf, explained_variance = split_rf_svd(strf[:, :, :, i])
            srfs.append(srf)
            trfs.append(trf)
            explained_variances.append(explained_variance)

        srf = np.stack(srfs, axis=-1)
        trf = np.stack(trfs, axis=-1)
        explained_variance = np.mean(explained_variances)

    else:
        raise NotImplementedError

    return srf, trf, explained_variance


def rescale_trf_srf(srf, trf, strf):
    # Rescale
    trf /= np.max(np.abs(trf))

    max_sign = np.sign(srf[np.unravel_index(np.argmax(np.abs(srf)), srf.shape)])
    srf *= (np.max(max_sign * strf) / np.max(max_sign * srf))

    merged_strf = merge_strf(srf=srf, trf=trf)
    if not np.isclose(np.max(max_sign * merged_strf), np.max(max_sign * strf), rtol=0.1):
        warnings.warn(f"{np.max(max_sign * merged_strf)} vs. {np.max(max_sign * strf)}")

    return srf, trf


def split_rf_sd(strf):
    """
    Project STRF onto sRF using standard deviation over time. Then get tRF from peak in this.
    """
    srf = np.std(strf, axis=0)
    ix, iy = np.unravel_index(np.argmax(np.abs(srf)), srf.shape)
    assert srf[ix, iy] == np.max(np.abs(srf)), (srf[ix, iy], np.max(np.abs(srf)))
    trf = strf[:, ix, iy]
    return srf, trf


def split_rf_peak(strf):
    """
    Find single peak in STRF and use to split into sRF and tRF.
    """
    it, ix, iy = np.unravel_index(np.argmax(np.abs(strf)), strf.shape)
    srf = strf[it]
    trf = strf[:, ix, iy]
    return srf, trf


def normalize_strf(strf):
    peak_sign = np.sign(strf[np.unravel_index(np.argmax(np.abs(strf)), strf.shape)])
    weight = np.sum(peak_sign * strf[(peak_sign * strf) > 0])
    norm_strf = strf / weight
    return norm_strf
