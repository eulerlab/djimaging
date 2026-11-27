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


def _bright_neighbour_field(seed=0):
    """make_field plus a bright cell (4x, own activity, no ROI) right of cell 3; ROI 3 drawn one pixel into it."""
    stack, mask = make_field(seed, shared=0.)
    rng = np.random.default_rng(seed + 100)
    xx, yy = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing='ij')
    foot = ndimage.gaussian_filter((np.hypot(xx - 19., yy - 14.) <= 2.3).astype(float), 0.6)
    stack = stack + 4 * foot[..., None] * (1 + _calcium(rng, stack.shape[2], 1.))[None, None]
    drawn = mask.copy()
    intrusion = [(19, 12)]  # cell 3: centre (19, 9), r 2.3 -> border at y = 11; bright cell border at y = 12
    for x, y in intrusion:
        drawn[x, y] = 3
    return stack, mask, drawn, intrusion


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_pixel_lit_by_a_bright_neighbour_is_removed_by_the_strict_default(seed):
    stack, mask, drawn, intrusion = _bright_neighbour_field(seed)
    shrunk, info = shrink_rois_to_fit(drawn, stack, n_artifact=2)
    assert info[3]['status'] == 'shrunk'
    assert set(info[3]['removed_bright']) == set(intrusion)
    assert all(shrunk[x, y] == 0 for x, y in intrusion)
    assert np.array_equal(shrunk[shrunk != drawn], np.zeros(len(intrusion)))  # nothing else
    shrunk, info = shrink_rois_to_fit(drawn, stack, n_artifact=2, bright_excess=None)  # gate disabled
    assert np.array_equal(shrunk, drawn)


def _blob(shape, centre, sigma, amp=100.):
    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    return amp * np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * sigma ** 2))


def _bright_px(roi, img, inside=None, min_excess=5., layer_frac=0.5):
    from djimaging.utils.mask_utils import _bright_border_pixels
    return _bright_border_pixels(roi, img, roi.copy() if inside is None else inside, min_excess, layer_frac)


def test_bright_gate_needs_light_from_outside():
    """A border pixel brighter than the inside is only removed if the brightness keeps rising outwards."""
    roi = np.zeros((15, 15), bool)
    roi[4:11, 4:11] = True
    img = 10 + _blob(roi.shape, (7, 7), 3.)
    img[10, 7] = 300.  # border pixel much brighter than its inner neighbours
    assert not _bright_px(roi, img).any()  # nothing brighter outside
    img[11, 7] = 400.  # bright structure just outside
    assert np.argwhere(_bright_px(roi, img)).tolist() == [[10, 7]]


def test_bright_gate_checks_more_layers_of_large_rois():
    roi = np.hypot(*np.meshgrid(np.arange(21) - 10., np.arange(21) - 10., indexing='ij')) <= 4.3  # 61 px: 2 layers
    small = np.hypot(*np.meshgrid(np.arange(21) - 10., np.arange(21) - 10., indexing='ij')) <= 2.3  # 1 layer
    for mask, n_expected in ((roi, 2), (small, 1)):
        img = 10 + _blob(mask.shape, (10, 10), 4.)
        x_out = np.flatnonzero(mask[:, 10]).max()  # outermost pixel along +x
        img[x_out - 1:x_out + 3, 10] = [150., 200., 250., 300.]  # brightness rising outwards over 2 ROI layers
        flagged = _bright_px(mask, img, min_excess=1.)
        expected = [[x, 10] for x in range(x_out - n_expected + 1, x_out + 1)]
        assert np.argwhere(flagged).tolist() == expected


def test_bright_gate_ignores_field_edge_and_ring_shaped_cells():
    # A cell cut off by the field edge (or the light artifact) is brightest at the cut: not a border there
    roi = np.zeros((12, 12), bool)
    roi[0:4, 3:9] = True
    img = 10 + _blob(roi.shape, (-1, 6), 2.5)
    inside = roi.copy()
    inside[0, :] = True  # beyond the edge counts as inside (as in shrink_rois_to_fit)
    img_edge = np.pad(img, ((1, 0), (0, 0)), constant_values=500.)  # outside the field: very bright (artifact)
    inside_edge = np.pad(inside, ((1, 0), (0, 0)), constant_values=True)
    assert not _bright_px(np.pad(roi, ((1, 0), (0, 0))), img_edge, inside_edge).any()
    # Nuclear exclusion: dark centre, bright ring inside a dim border
    roi = np.hypot(*np.meshgrid(np.arange(17) - 8., np.arange(17) - 8., indexing='ij')) <= 4.3
    r = np.hypot(*np.meshgrid(np.arange(17) - 8., np.arange(17) - 8., indexing='ij'))
    img = 10 + 100 * np.exp(-(r - 2.5) ** 2 / 2.)
    assert not _bright_px(roi, img).any()


def test_bright_gate_with_real_shapes_at_the_light_artifact():
    """shrink_rois_to_fit: a ROI touching very bright light artifact rows is not shrunk towards them."""
    stack, mask = make_field(0, n_frames=300)
    stack[:2] += 50.  # light artifact
    mask[mask == 1] = 0
    mask[2:5, 6:11] = 1  # ROI right below the artifact
    shrunk, info = shrink_rois_to_fit(mask, stack, n_artifact=2)
    assert info[1]['n_bright'] == 0


def test_input_checks():
    stack, mask = make_field(0, n_frames=200)
    with pytest.raises(ValueError, match='python format'):
        shrink_rois_to_fit(-mask, stack)
    with pytest.raises(ValueError, match='shape'):
        shrink_rois_to_fit(mask[:-1], stack)
