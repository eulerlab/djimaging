"""shrink_rois_to_fit: remove border pixels that do not fit the activity of their ROI, leave everything else."""
import numpy as np
import pytest
from scipy import ndimage

from djimaging.utils.mask_utils import SHRINK_SENSITIVE, shrink_rois_to_fit


def _calcium(rng, n, rate, fs=7.8, tau=1.):
    kernel = np.exp(-np.arange(int(5 * tau * fs)) / (tau * fs))
    return np.convolve(rng.poisson(rate / fs, n).astype(float), kernel)[:n]


def make_field(seed=0, n_frames=1500, noise=0.5, vessel=False, shared=0.3):
    """Three cells (discs r=2.3 px, blurred), own activity, smooth neuropil, shot noise. Cells 1 and 2 touch.
    With ``vessel``, a dark vertical stripe (no neuropil, no cell) lies right next to cell 3."""
    rng = np.random.default_rng(seed)
    nx, ny = 26, 26
    centers = [(10., 8.), (10., 13.), (19., 9.)]
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    discs = [np.hypot(xx - cx, yy - cy) <= 2.3 for cx, cy in centers]
    dark = np.zeros((nx, ny), bool)
    if vessel:
        dark[14:25, 12:14] = True  # just right of cell 3 (its border is at y = 11)
    common = _calcium(rng, n_frames, 0.6)
    movie = 0.35 * (1 + 0.5 * common + _calcium(rng, n_frames, 1.))[None, None] * \
        (0.7 + 0.3 * ndimage.gaussian_filter(rng.random((nx, ny)), 4))[..., None] * ~dark[..., None]
    for disc in discs:
        foot = ndimage.gaussian_filter(disc.astype(float), 0.6) * ~dark
        movie = movie + foot[..., None] * (1 + shared * common + (1 - shared) * _calcium(rng, n_frames, 0.4))[None, None]
    movie = movie + noise * np.sqrt(np.maximum(movie, 0.05)) * rng.standard_normal(movie.shape)
    mask = np.zeros((nx, ny), dtype=int)
    for i, disc in enumerate(discs, start=1):
        mask[disc & (mask == 0)] = i
    return movie, mask


PRESETS = dict(strict=dict(), sensitive=SHRINK_SENSITIVE)


@pytest.mark.parametrize('preset', PRESETS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_correct_rois_are_unchanged(seed, preset):
    stack, mask = make_field(seed)
    shrunk, info = shrink_rois_to_fit(mask, stack, n_artifact=2, **PRESETS[preset])
    assert np.array_equal(shrunk, mask)
    assert all(d['status'] == 'unchanged' for d in info.values())


def _drawn_into_neighbour(mask):
    ring = ndimage.binary_dilation(mask == 1, ndimage.generate_binary_structure(2, 1)) & (mask == 2)
    drawn = mask.copy()
    drawn[ring] = 1  # ROI 1 drawn one pixel into cell 2
    return drawn, ring


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_pixels_of_a_neighbouring_cell_are_removed_by_the_sensitive_preset(seed):
    stack, mask = make_field(seed, shared=0.)  # neighbour with its own activity
    drawn, ring = _drawn_into_neighbour(mask)
    shrunk, info = shrink_rois_to_fit(drawn, stack, n_artifact=2, **SHRINK_SENSITIVE)
    assert info[1]['status'] == 'shrunk'
    removed = (drawn == 1) & (shrunk != 1)
    assert removed.sum() >= 0.5 * ring.sum() and not np.any(removed & (mask == 1))  # only wrong pixels
    assert info[2]['status'] == 'unchanged' and info[3]['status'] == 'unchanged'


def test_strict_default_leaves_mixed_border_pixels():
    """One pixel between two touching cells is mixed (blur, shared activity): the strict default keeps it."""
    stack, mask = make_field(0)
    drawn, _ = _drawn_into_neighbour(mask)
    shrunk, _ = shrink_rois_to_fit(drawn, stack, n_artifact=2)
    assert np.array_equal(shrunk, drawn)


@pytest.mark.parametrize('seed', [0, 1])
def test_pixels_without_signal_are_removed_by_the_sensitive_preset(seed):
    stack, mask = make_field(seed, vessel=True)
    drawn = mask.copy()
    band = ndimage.shift(mask == 3, (0, 1), order=0) & (mask == 0)  # one pixel too large, into the dark stripe
    drawn[band] = 3
    shrunk, info = shrink_rois_to_fit(drawn, stack, n_artifact=2, **SHRINK_SENSITIVE)
    assert info[3]['status'] == 'shrunk'
    removed = (drawn == 3) & (shrunk != 3)
    # the corner pixels of the band still see the cell (blur) and are kept
    assert removed.sum() >= 0.5 * band.sum() and not np.any(removed & (mask == 3))


def test_never_adds_pixels_keeps_labels_and_respects_roi_ids():
    stack, mask = make_field(0)
    drawn = mask.copy()
    drawn[ndimage.binary_dilation(mask == 1, ndimage.generate_binary_structure(2, 1)) & (mask == 2)] = 1
    shrunk, info = shrink_rois_to_fit(drawn, stack, n_artifact=2, roi_ids=[2, 3])
    assert np.array_equal(shrunk, drawn) and set(info) == {2, 3}  # ROI 1 not checked
    shrunk, _ = shrink_rois_to_fit(drawn, stack, n_artifact=2)
    assert np.all((shrunk == drawn) | (shrunk == 0))
    assert set(np.unique(shrunk)) == set(np.unique(drawn))


def test_small_and_silent_rois_are_left_alone():
    stack, mask = make_field(0)
    drawn = mask.copy()
    drawn[2:4, 20:22] = 7  # 4 px
    drawn[22:26, 20:24] = 8  # background only
    _, info = shrink_rois_to_fit(drawn, stack, n_artifact=2)
    assert info[7]['status'] == 'too_small'
    assert info[8]['status'] in ('low_signal', 'unchanged')


def test_input_checks():
    stack, mask = make_field(0, n_frames=200)
    with pytest.raises(ValueError, match='python format'):
        shrink_rois_to_fit(-mask, stack)
    with pytest.raises(ValueError, match='shape'):
        shrink_rois_to_fit(mask[:-1], stack)
