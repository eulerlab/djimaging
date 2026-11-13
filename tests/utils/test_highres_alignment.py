"""Unit tests for djimaging.utils.highres_alignment on synthetic fields with a known HR shift."""
import numpy as np
import pytest
from scipy import ndimage

from djimaging.utils.highres_alignment import align_highres, resample_to_field

FIELD_PX, SCALE, N = 1.5, 8, 64


def _scene(n_hr, seed=0, n_cells=60, cell_sigma_um=2.5, n_vessels=4):
    """Cells (Gaussian blobs) and vessels (lines) on a fine grid of n_hr x n_hr px of FIELD_PX / SCALE um."""
    rng = np.random.default_rng(seed)
    px = FIELD_PX / SCALE
    x = np.arange(n_hr) * px
    cells = np.zeros((n_hr, n_hr))
    for cx, cy in rng.uniform(5, n_hr * px - 5, (n_cells, 2)):
        cells += np.exp(-((x[:, None] - cx) ** 2 + (x[None, :] - cy) ** 2) / (2 * cell_sigma_um ** 2))
    vessels = np.zeros((n_hr, n_hr))
    for _ in range(n_vessels):
        a, b = rng.uniform(0, n_hr * px, 2)
        ang = rng.uniform(0, np.pi)
        d = np.abs((x[:, None] - a) * np.sin(ang) - (x[None, :] - b) * np.cos(ang))
        vessels += np.exp(-d ** 2 / (2 * 1.5 ** 2))
    return 100 + 50 * cells, 100 + 50 * vessels


def _cells_mask(field_img, thresh=0.6):
    lab, n = ndimage.label(field_img > np.percentile(field_img, 100 * thresh) + 0 * field_img)
    mask = np.ones(field_img.shape, int)
    for i in range(1, n + 1):
        mask[lab == i] = -i
    return mask


@pytest.fixture(scope='module')
def scene():
    n_hr = N * SCALE
    c, v = _scene(n_hr + 2 * 6 * SCALE)  # larger canvas so that shifted HR scans stay inside
    return c, v


def _hr_and_field(scene, shift_field_px, n_frames=4, seed=1):
    """Field mean images (64 x 64) and an HR stack (512 x 512 x t) whose content is displaced by ``shift_field_px``."""
    c, v = scene
    pad = 6 * SCALE
    n_hr = N * SCALE
    sl = slice(pad, pad + n_hr)
    rng = np.random.default_rng(seed)
    field = [resample_to_field(img[sl, sl], FIELD_PX / SCALE, (N, N), FIELD_PX) for img in (c, v)]
    s = np.asarray(shift_field_px) * SCALE  # content at p is at p + shift in the HR -> sample at p - shift
    hr = [ndimage.shift(img, s, order=1)[sl, sl] for img in (c, v)]
    hr = [np.stack([h + rng.normal(0, 2, h.shape) for _ in range(n_frames)], axis=2) for h in hr]
    return field, hr


@pytest.mark.parametrize('shift', [(0., 0.), (1.4, -0.6), (-2.25, 3.1)])
def test_align_highres_recovers_shift_and_matches(scene, shift):
    (f0, f1), (h0, h1) = _hr_and_field(scene, shift)
    mask = _cells_mask(f0)
    a = align_highres(h0, h1, FIELD_PX / SCALE, 0, f0, f1, FIELD_PX, 0, roi_mask=mask)
    assert np.allclose(a['shift'], shift, atol=0.1)
    assert a['scale'] == pytest.approx(SCALE)
    assert a['match_ch0'] > 0.95 and a['match_ch1'] > 0.95
    assert a['dz_um_approx'] < 1.5
    assert a['hr_ch0'].shape == (N, N)
    assert a['mask_fit_hr'] > 0.5 * a['mask_fit_ref']
    # 2D input (channel medians, as stored in HighRes.StackAverages) gives the same shift
    b = align_highres(np.median(h0, 2), np.median(h1, 2), FIELD_PX / SCALE, 0, f0, f1, FIELD_PX, 0)
    assert np.allclose(b['shift'], a['shift'], atol=0.05)
    assert b['match_ch0'] == pytest.approx(a['match_ch0'], abs=0.03)
    assert b['match_ch1'] == pytest.approx(a['match_ch1'], abs=0.03)
    assert np.isfinite(b['dz_um_approx'])


def test_align_highres_match_drops_for_other_plane(scene):
    (f0, f1), (h0, h1) = _hr_and_field(scene, (0.5, 0.5))
    blurred = [ndimage.gaussian_filter(h, (2.5 * SCALE, 2.5 * SCALE, 0)) for h in (h0, h1)]  # strong defocus
    a = align_highres(*blurred, FIELD_PX / SCALE, 0, f0, f1, FIELD_PX, 0)
    good = align_highres(h0, h1, FIELD_PX / SCALE, 0, f0, f1, FIELD_PX, 0)
    assert a['match_ch1'] < good['match_ch1'] - 0.1
    assert a['dz_um_approx'] > good['dz_um_approx']


def test_resample_to_field_block_mean_and_offset():
    hr = np.arange(16 * 16, dtype=float).reshape(16, 16)
    out = resample_to_field(hr, 1., (4, 4), 4.)
    assert out[1, 2] == pytest.approx(hr[4:8, 8:12].mean())
    # offset of one field pixel = sampling 4 HR px further; the border that leaves the HR FOV is NaN
    out2 = resample_to_field(hr, 1., (4, 4), 4., offset=(1., 0.))
    assert np.allclose(out2[:3], out[1:]) and np.all(np.isnan(out2[3]))
