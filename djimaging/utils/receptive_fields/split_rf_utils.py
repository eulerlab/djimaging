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

    if method.lower() == 'svd':
        srf, trf = split_rf_svd(strf)
    elif method.lower() == 'sd':
        srf, trf = split_rf_sd(strf)
    elif method.lower() == 'peak':
        srf, trf = split_rf_peak(strf)
    else:
        raise NotImplementedError(f"Method {method} for RF split not implemented")

    if upsample_srf_scale > 1:
        srf = resize_srf(srf, scale=upsample_srf_scale)

    return srf, trf


def split_rf_svd(strf):
    """
    Assuming an RF is time-space separable, get spatial and temporal filters using SVD.
    From RFEst.
    """
    dims = strf.shape

    if len(dims) == 3:
        dims_tRF = dims[0]
        dims_sRF = dims[1:]
        U, S, Vt = randomized_svd(strf.reshape(dims_tRF, -1), 3, random_state=0)
        srf = Vt[0].reshape(*dims_sRF)
        trf = U[:, 0]

    elif len(dims) == 2:
        dims_tRF = dims[0]
        dims_sRF = dims[1]
        U, S, Vt = randomized_svd(strf.reshape(dims_tRF, dims_sRF), 3, random_state=0)
        srf = Vt[0]
        trf = U[:, 0]

    else:
        raise NotImplementedError

    srf, trf = rescale_trf_srf(srf, trf, strf)
    return srf, trf


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