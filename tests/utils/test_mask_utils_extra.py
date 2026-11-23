"""Hand-checked tests of the ROI mask helpers in mask_utils (non-square masks to catch axis swaps)."""
import os

import matplotlib
import numpy as np
import pytest

from djimaging.utils import mask_utils as mu

matplotlib.use('Agg')


@pytest.fixture
def no_show(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    yield
    plt.close('all')


# ---------------------------------------------------------------- files

def test_file_format_helpers():
    assert mu._infer_roi_file_format('a/b_ROIs.npy') == 'numpy'
    assert mu._infer_roi_file_format('a/b_ROIs.pkl') == 'pickle'
    with pytest.raises(ValueError, match='Cannot infer'):
        mu._infer_roi_file_format('a/b_ROIs.h5')
    assert mu._get_roi_mask_suffix('numpy') == '_ROIs.npy'
    assert mu._get_roi_mask_suffix('pickle') == '_ROIs.pkl'
    with pytest.raises(ValueError, match='Unknown file_format'):
        mu._get_roi_mask_suffix('h5')


@pytest.mark.parametrize('file_format, ext', [('numpy', '.npy'), ('pickle', '.pkl')])
def test_save_load_roi_mask_round_trip(tmp_path, file_format, ext):
    mask = np.array([[0, 1, 1], [2, 0, 0]], dtype=np.int32)
    path = str(tmp_path / f'rec_ROIs{ext}')
    mu.save_roi_mask_file(path, mask, file_format)
    loaded = mu.load_roi_mask_file(path)
    np.testing.assert_array_equal(loaded, mask)
    assert loaded.dtype == mask.dtype


def test_save_load_roi_mask_bad_format(tmp_path):
    with pytest.raises(ValueError, match='Unknown file_format'):
        mu.save_roi_mask_file(str(tmp_path / 'x.npy'), np.zeros((2, 2)), 'h5')
    with pytest.raises(ValueError, match='Cannot infer'):
        mu.load_roi_mask_file(str(tmp_path / 'x.h5'))


def test_to_roi_mask_file_variants():
    data_file = os.path.join('root', 'Pre', 'SMP_rec.h5')
    assert mu.to_roi_mask_file(data_file, file_format='pickle') == os.path.join('root', 'Pre', 'SMP_rec_ROIs.pkl')
    assert mu.to_roi_mask_file(data_file, new_prefix='X_') == os.path.join('root', 'Pre', 'X_SMP_rec_ROIs.npy')
    # prefix already present: not added twice
    assert mu.to_roi_mask_file(data_file, new_prefix='SMP_') == os.path.join('root', 'Pre', 'SMP_rec_ROIs.npy')
    assert mu.to_roi_mask_file(data_file, old_prefix='SMP_', new_prefix='M_', roi_mask_dir='Masks') == \
        os.path.join('root', 'Masks', 'M_rec_ROIs.npy')
    assert mu.to_roi_mask_file(os.path.join('r', 'Raw', 'rec.smp'), old_suffix='smp', roi_mask_dir='Pre') == \
        os.path.join('r', 'Pre', 'rec_ROIs.npy')
    assert mu.to_field_roi_mask_file(data_file) == os.path.join('root', 'Pre', 'SMP_rec_FieldROIs.npy')
    with pytest.raises(ValueError):
        mu.to_roi_mask_file(data_file, file_format='h5')


def test_load_preferred_roi_mask_file(tmp_path):
    files = [str(tmp_path / 'Pre' / f'f1_{s}.h5') for s in ('chirp', 'mb')]
    os.makedirs(tmp_path / 'Pre')
    mask = np.array([[0, 2], [1, 0], [0, 0]])
    mu.save_roi_mask_file(str(tmp_path / 'Pre' / 'f1_mb_ROIs.pkl'), mask, 'pickle')
    loaded, path = mu.load_preferred_roi_mask_file(files, file_format='pickle')
    np.testing.assert_array_equal(loaded, [[1, -2], [-1, 1], [1, 1]])
    assert path == str(tmp_path / 'Pre' / 'f1_mb_ROIs.pkl')
    assert mu.load_preferred_roi_mask_file(files, file_format='numpy') == (None, None)
    with pytest.raises(ValueError, match='file_format is not set'):
        mu.load_preferred_roi_mask_file(files, file_format=None)


def test_load_preferred_roi_mask_igor_without_h5(tmp_path):
    other = tmp_path / 'f1_chirp.smp'
    other.write_bytes(b'')
    assert mu.load_preferred_roi_mask_igor([str(other), str(tmp_path / 'missing.h5')]) == (None, None)


def test_sort_roi_mask_files_alias_and_control():
    files = ['/x/f1_chirp_control.h5', '/x/f1_dn_control.h5', '/x/f1_mb_control.h5']
    assert list(mu.sort_roi_mask_files(files, mask_alias='mb_dn')) == [files[2], files[1], files[0]]
    np.testing.assert_array_equal(mu.sort_roi_mask_files(files, mask_alias='mb_dn', as_index=True), [2, 1, 0])
    # control conditions win over the alias order
    files = ['/x/f1_mb_drug.h5', '/x/f1_dn_control.h5']
    assert list(mu.sort_roi_mask_files(files, mask_alias='mb_dn')) == [files[1], files[0]]
    # high resolution recordings come last (before the non-control penalty)
    files = ['/x/f1_hr_control.h5', '/x/f1_chirp_control.h5', '/x/f1_mb_control.h5']
    assert list(mu.sort_roi_mask_files(files, mask_alias='mb', highres_alias='hr')) == [files[2], files[1], files[0]]
    # case insensitive
    files = ['/x/F1_CHIRP_C1.h5', '/x/F1_MB_C1.h5']
    assert list(mu.sort_roi_mask_files(files, mask_alias='mb')) == [files[1], files[0]]


def test_sort_roi_mask_files_ties_alphabetical():
    files = ['/x/c_control.h5', '/x/a_control.h5', '/x/b_control.h5']
    assert list(mu.sort_roi_mask_files(files)) == ['/x/a_control.h5', '/x/b_control.h5', '/x/c_control.h5']


# ---------------------------------------------------------------- drawing helpers

def test_create_circular_mask_shape_and_orientation():
    mask = mu.create_circular_mask(h=3, w=5, center=(4, 0), radius=1)
    assert mask.shape == (5, 3)  # (w, h)
    expected = np.zeros((5, 3), dtype=bool)
    expected[[3, 4, 4], [0, 0, 1]] = True
    np.testing.assert_array_equal(mask, expected)
    assert mu.create_circular_mask(h=4, w=6, center=(2, 1), radius=0).sum() == 1


def test_extract_connected_mask_four_connectivity():
    # no foreground in the last row and column (see test_extract_connected_mask_no_wrap_around)
    mask = np.array([[1, 1, 0, 0, 0],
                     [0, 1, 0, 1, 0],
                     [0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0]])
    out = mu.extract_connected_mask(mask, 1, 1)
    expected = np.zeros_like(mask)
    expected[[0, 0, 1, 2, 2], [0, 1, 1, 1, 2]] = 1
    np.testing.assert_array_equal(out, expected)  # diagonal (1, 3) not connected
    np.testing.assert_array_equal(mask[0], [1, 1, 0, 0, 0])  # input unchanged
    # seed on background: empty mask
    assert mu.extract_connected_mask(mask, 3, 0).sum() == 0
    # isolated pixel
    np.testing.assert_array_equal(np.argwhere(mu.extract_connected_mask(mask, 1, 3)), [[1, 3]])
    # seed at the far corner (border pixels at max index)
    corner = np.zeros((4, 5), dtype=int)
    corner[2:4, 4] = 1
    np.testing.assert_array_equal(np.argwhere(mu.extract_connected_mask(corner, 3, 4)), [[2, 4], [3, 4]])


@pytest.mark.parametrize('pixels, seed', [
    (((0, 1), (3, 1)), (0, 1)),  # first and last row
    (((1, 0), (1, 4)), (1, 0)),  # first and last column
])
def test_extract_connected_mask_no_wrap_around(pixels, seed):
    mask = np.zeros((4, 5), dtype=int)
    for p in pixels:
        mask[p] = 1
    out = mu.extract_connected_mask(mask, *seed)
    np.testing.assert_array_equal(np.argwhere(out), [seed])


def test_get_mask_by_bg_non_square():
    data = np.zeros((5, 8))
    data[1:3, 2:5] = 1.
    data[1, 6] = 1.  # same value but not connected
    out = mu.get_mask_by_bg(1, 2, data, thresh=0.2, max_pixel_dist=10)
    np.testing.assert_array_equal(out, data * (np.arange(8) < 6)[None, :])
    # max_pixel_dist limits the mask around the seed
    out = mu.get_mask_by_bg(1, 2, data, max_pixel_dist=1)
    assert out.sum() == 3 and out[1, 2] and out[2, 2] and out[1, 3]
    # explicit reference value: seed pixel itself is too far away
    assert mu.get_mask_by_bg(1, 2, data, ref_value=0.5, thresh=0.2).sum() == 0


def test_get_mask_by_bg_no_wrap_around():
    data = np.zeros((4, 6))
    data[0, 2] = data[3, 2] = 1.
    out = mu.get_mask_by_bg(0, 2, data, thresh=0.2, max_pixel_dist=10)
    np.testing.assert_array_equal(np.argwhere(out), [[0, 2]])


def test_get_mask_by_bg_plot(no_show):
    data = np.zeros((4, 6))
    data[1:3, 1:3] = 1.
    assert mu.get_mask_by_bg(1, 1, data, plot=True).sum() == 4


def _cc_data(rng, shape=(6, 9), n_t=200):
    signal = rng.normal(size=n_t)
    data = rng.normal(size=shape + (n_t,))
    roi = np.zeros(shape, dtype=bool)
    roi[2:4, 3:7] = True
    data[roi] += 5 * signal
    return data, roi, signal


def test_get_mask_by_cc():
    data, roi, signal = _cc_data(np.random.default_rng(0))
    np.testing.assert_array_equal(mu.get_mask_by_cc(2, 3, data, thresh=0.5).astype(bool), roi)
    np.testing.assert_array_equal(mu.get_mask_by_cc(2, 3, data, seed_trace=signal, thresh=0.5).astype(bool), roi)
    out = mu.get_mask_by_cc(2, 3, data, thresh=0.5, max_pixel_dist=1)
    assert out.sum() == 3


def test_get_mask_by_cc_plot(no_show):
    data, roi, _ = _cc_data(np.random.default_rng(1))
    np.testing.assert_array_equal(mu.get_mask_by_cc(3, 6, data, thresh=0.5, plot=True).astype(bool), roi)


# ---------------------------------------------------------------- cleaning

def _labels():
    m = np.zeros((4, 7), dtype=int)
    m[0, 0:2] = 3
    m[0, 5:7] = 3  # ROI 3 disconnected
    m[3, 3] = 2  # single pixel
    m[2:4, 0] = 1
    return m


def test_relabel_and_fix_disconnected_rois(capsys):
    m = _labels()
    new, n = mu.fix_disconnected_rois(m, verbose=True, return_num=True)
    assert n == 4
    assert 'ROI#3 is disconnected into 2 parts' in capsys.readouterr().out
    # labels are numbered in column-major order (label of the transposed mask)
    expected = np.zeros_like(m)
    expected[0, 0:2] = 1
    expected[2:4, 0] = 2
    expected[3, 3] = 3
    expected[0, 5:7] = 4
    np.testing.assert_array_equal(new, expected)
    np.testing.assert_array_equal(mu.fix_disconnected_rois(m, verbose=False), expected)
    np.testing.assert_array_equal(mu.relabel_mask(m, connectivity=1), expected)


def test_relabel_connectivity():
    m = np.array([[1, 0, 0], [0, 1, 0]])
    assert mu.relabel_mask(m, connectivity=2, return_num=True)[1] == 1
    assert mu.relabel_mask(m, connectivity=1, return_num=True)[1] == 2


def test_remove_small_rois(capsys):
    out = mu.remove_small_rois(_labels(), min_size=2)
    expected = _labels()
    expected[3, 3] = 0
    np.testing.assert_array_equal(out, expected)
    assert '1 ROIs <2px removed' in capsys.readouterr().out
    np.testing.assert_array_equal(mu.remove_small_rois(_labels(), min_size=1, verbose=True), _labels())
    assert capsys.readouterr().out == ''


def test_shrink_light_artifact_rois(capsys):
    out = mu.shrink_light_artifact_rois(_labels(), n_artifact=1)
    expected = _labels()
    expected[0] = 0
    np.testing.assert_array_equal(out, expected)
    assert '1 ROIs in light artifact region' in capsys.readouterr().out
    np.testing.assert_array_equal(mu.shrink_light_artifact_rois(_labels(), n_artifact=0), _labels())


def test_clean_rois():
    out = mu.clean_rois(_labels(), n_artifact=1, min_size=2, verbose=False)
    expected = np.zeros((4, 7), dtype=int)
    expected[2:4, 0] = 1
    np.testing.assert_array_equal(out, expected)
    out = mu.clean_rois(_labels(), n_artifact=0, min_size=2, verbose=False)
    assert len(np.unique(out[out > 0])) == 3  # ROI 3 split into two, ROI 2 removed


# ---------------------------------------------------------------- vessels

def test_robust_z():
    img = np.array([[100., 100.], [1., 2.], [3., 4.]])
    z = mu._robust_z(img, n_artifact=1)
    med, mad = 2.5, 1.4826 * 1.
    np.testing.assert_allclose(z, (img - med) / mad)
    np.testing.assert_array_equal(mu._robust_z(np.ones((3, 2))), np.zeros((3, 2)))
    # MAD zero: falls back to std
    img = np.array([0., 0., 0., 0., 4.]).reshape(5, 1)
    np.testing.assert_allclose(mu._robust_z(img), img / np.std(img))


def _vessel_images(shape=(30, 20)):
    rng = np.random.default_rng(0)
    main = rng.normal(1, 0.05, shape)
    alt = 0.5 * main + rng.normal(0, 0.05, shape)
    main[5:8, 4:7] += 3.  # a cell, bright in both channels (bleed-through)
    alt[5:8, 4:7] += 1.5
    alt[15:25, 12:14] += 20.  # a vessel, only in the alternative channel
    return alt, main


def test_estimate_vessel_mask():
    alt, main = _vessel_images()
    vessel = mu.estimate_vessel_mask(alt, main, n_artifact=2, alt_z=20)
    assert vessel.shape == (30, 20) and vessel.dtype == bool
    assert vessel[16:24, 12:14].all()
    assert not vessel[5:8, 4:7].any()
    assert not vessel[:2].any()
    assert vessel.sum() <= 2 * 10 * 4  # vessel plus some smoothing spill-over
    # no smoothing
    vessel = mu.estimate_vessel_mask(alt, main, alt_z=20, smooth_sigma=0)
    assert vessel[15:25, 12:14].all() and not vessel[5:8, 4:7].any()
    with pytest.raises(ValueError, match='same shape'):
        mu.estimate_vessel_mask(alt, main.T)


def test_remove_vessel_rois(capsys):
    mask = np.zeros((5, 6), dtype=int)
    mask[0:2, 0:2] = 4  # 1 of 4 px on vessel: kept with max_overlap 0.3
    mask[3:5, 3:6] = 7  # 3 of 6 px on vessel: removed
    vessel = np.zeros((5, 6), dtype=bool)
    vessel[0, 0] = True
    vessel[3:5, 5] = True
    vessel[3, 4] = True
    out = mu.remove_vessel_rois(mask, vessel, max_overlap=0.3)
    expected = np.zeros_like(mask)
    expected[0:2, 0:2] = 1
    np.testing.assert_array_equal(out, expected)
    assert 'Removed 1 ROIs' in capsys.readouterr().out
    with pytest.raises(ValueError, match='same shape'):
        mu.remove_vessel_rois(mask, vessel.T)


# ---------------------------------------------------------------- adding ROIs

def test_find_neighbor_labels():
    labels = np.arange(6).reshape(2, 3)  # (y, x)
    # [north (y+1), south (y-1), east (x+1), west (x-1)], 0 outside
    np.testing.assert_array_equal(mu.find_neighbor_labels(0, 0, labels), [3, 0, 1, 0])
    np.testing.assert_array_equal(mu.find_neighbor_labels(2, 1, labels), [0, 2, 0, 4])
    np.testing.assert_array_equal(mu.find_neighbor_labels(1, 0, labels), [4, 0, 2, 0])


def _rois_and_suggestions():
    rois = np.zeros((6, 9), dtype=int)
    rois[1:3, 1:3] = 1
    add = np.zeros((6, 9), dtype=int)
    add[1:3, 2:6] = 5  # overlaps (column 2) and touches (column 3) ROI 1
    add[4:6, 6:9] = 7  # free
    add[0, 8] = 9  # single pixel: too small
    return rois, add


def test_add_rois():
    rois, add = _rois_and_suggestions()
    out, new = mu.add_rois(add, rois)
    expected = np.zeros_like(rois)
    expected[1:3, 1:3] = 1
    expected[1:3, 4:6] = 2
    expected[4:6, 6:9] = 3
    np.testing.assert_array_equal(out, expected)
    expected_new = np.zeros_like(rois)
    expected_new[1:3, 4:6] = 1
    expected_new[4:6, 6:9] = 2
    np.testing.assert_array_equal(new, expected_new)
    np.testing.assert_array_equal(rois[1:3, 1:3], 1)  # input unchanged
    assert not np.any(rois[:, 3:])


def test_add_rois_splits_and_plots(no_show):
    rois = np.zeros((5, 9), dtype=int)
    rois[0:2, 4] = 1
    add = np.zeros((5, 9), dtype=int)
    add[0:2, 0:9] = 2  # the touching/overlapping part of column 3..5 is removed, the ROI splits in two
    out, new = mu.add_rois(add, rois, connectivity=2, plot=True)
    assert len(np.unique(new[new > 0])) == 2
    assert np.all(out[0:2, 3:6][rois[0:2, 3:6] == 0] == 0)
    assert np.all(new[0:2, 0:3] > 0) and np.all(new[0:2, 6:9] > 0)


def test_generate_roi_suggestions():
    true = np.zeros((6, 8), dtype=int)
    true[1:3, 1:3] = 1
    pred = np.zeros((6, 8), dtype=int)
    pred[1:3, 1:4] = 4  # IoU 4/6 with the true ROI: not a suggestion
    pred[3:6, 5:8] = 2  # new
    pred[0, 7] = 3  # in the artifact row: removed
    out = mu.generate_roi_suggestions(pred, true, n_artifact=1)
    expected = np.zeros_like(true)
    expected[3:6, 5:8] = 2  # after relabelling: first ROI in column-major order is the matched one
    np.testing.assert_array_equal(out, expected)
    # a low IoU threshold of 0 never suggests anything
    assert not np.any(mu.generate_roi_suggestions(pred, true, n_artifact=1, threshold=0.))


def test_generate_roi_suggestions_empty_true_mask():
    pred = np.zeros((4, 5), dtype=int)
    pred[1:4, 1:4] = 1
    out = mu.generate_roi_suggestions(pred, np.zeros((4, 5), dtype=int), n_artifact=0)
    np.testing.assert_array_equal(out, pred)


# ---------------------------------------------------------------- shifting and comparing

def test_shift_array_values_non_square():
    a = np.arange(20).reshape(4, 5)
    out = mu.shift_array(a, (1, -2), cval=-1)
    expected = np.full((4, 5), -1)
    expected[1:, :3] = a[:3, 2:]
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(a, np.arange(20).reshape(4, 5))  # copy
    # default cval: min of the image
    assert mu.shift_array(a + 3, (0, 1))[0, 0] == 3
    assert mu.shift_array(a.astype(float), (2, 0), cval=np.median)[0, 0] == np.median(a)
    # shifts beyond the image: only fill
    assert np.all(mu.shift_array(a, (5, 0), cval=-1) == -1)
    assert np.all(mu.shift_array(a, (0, -6), cval=-1) == -1)
    # round trip of the inner part
    back = mu.shift_array(mu.shift_array(a, (1, -2), cval=-1), (-1, 2), cval=-1)
    np.testing.assert_array_equal(back[:3, 2:], a[:3, 2:])


def test_shift_array_3d_stack():
    stack = np.arange(24).reshape(3, 4, 2)
    out = mu.shift_array(stack, (0, 1), cval=-1)
    np.testing.assert_array_equal(out[:, 1:], stack[:, :-1])
    assert np.all(out[:, 0] == -1)


def test_shift_array_light_artifact():
    a = np.arange(20).reshape(4, 5)
    out = mu.shift_array(a, (1, 0), cval=-1, n_artifact=1)
    expected = np.full((4, 5), -1)
    expected[2:] = a[1:3]
    np.testing.assert_array_equal(out, expected)
    out = mu.shift_array(a, (-1, 0), cval=-1, n_artifact=2)
    expected = np.full((4, 5), -1)
    expected[1:3] = a[2:4]
    np.testing.assert_array_equal(out, expected)


def test_shift_array_inplace():
    a = np.arange(20).reshape(4, 5)
    out = mu.shift_array(a, (1, 0), inplace=True, cval=-1)
    np.testing.assert_array_equal(a, out)


def _igor_ref(shape=(32, 24)):
    ref = np.ones(shape, dtype=int)
    ref[10:13, 8:11] = -1
    ref[20:23, 14:16] = -2
    ref[5:7, 18:20] = -3
    return ref


def test_compare_roi_masks_same_different_shifted():
    ref = _igor_ref()
    assert mu.compare_roi_masks(ref.copy(), ref) == ('same', (0, 0))
    assert mu.compare_roi_masks(ref[:30], ref) == ('different', (0, 0))
    status, shift = mu.compare_roi_masks(mu.shift_array(ref, (1, -2), cval=1), ref)
    assert (status, shift) == ('shifted', (-1, 2))
    # a ROI cut at the border by the shift is still a shifted copy
    ref2 = np.ones((32, 24), dtype=int)
    ref2[0:3, 8:11] = -1
    ref2[20:23, 14:16] = -2
    assert mu.compare_roi_masks(mu.shift_array(ref2, (-1, 0), cval=1), ref2) == ('shifted', (1, 0))
    # an additional ROI: different
    extra = ref.copy()
    extra[0:2, 0:2] = -4
    assert mu.compare_roi_masks(extra, ref) == ('different', (0, 0))
    # shift larger than max_shift: different
    assert mu.compare_roi_masks(mu.shift_array(ref, (3, 0), cval=1), ref, max_shift=2) == ('different', (0, 0))
    with pytest.raises(ValueError):
        mu.compare_roi_masks(ref + 1, ref)  # not Igor format


def test_compare_roi_masks_empty_mask_is_different():
    assert mu.compare_roi_masks(np.ones((32, 24), dtype=int), _igor_ref()) == ('different', (0, 0))


def test_compare_roi_masks_rois_near_edge_true_shift():
    ref = np.ones((32, 24), dtype=int)
    ref[10:13, 0:3] = -1
    ref[20:23, 1:4] = -2  # all ROIs within 5 px of the y = 0 edge
    roi_mask = mu.shift_array(ref, (2, 0), cval=1)
    assert mu.compare_roi_masks(roi_mask, ref) == ('shifted', (-2, 0))


def test_offset_image():
    img = np.arange(12, dtype=float).reshape(3, 4)
    out = mu._offset_image(img, (1, -1))
    # out[p] = img[p - offset]
    assert np.isnan(out[0]).all() and np.isnan(out[:, -1]).all()
    np.testing.assert_array_equal(out[1:, :-1], img[:-1, 1:])
    np.testing.assert_array_equal(mu._offset_image(img, (0, 0)), img)


def test_best_mask_offset_without_rois():
    img = np.random.default_rng(0).normal(size=(20, 16))
    res = mu.best_mask_offset(np.ones((20, 16), dtype=int), img)
    assert res['offset'] == (0, 0) and np.isnan(res['score'])
    assert np.isnan(mu.mask_fit_score(np.ones((20, 16), dtype=int), img))


def test_select_field_roi_mask_errors_and_alias_fallback():
    mask = np.ones((20, 16), dtype=int)
    img = np.random.default_rng(0).normal(size=(20, 16))
    with pytest.raises(ValueError, match='2 ROI masks but 1 images'):
        mu.select_field_roi_mask([mask, mask], [img])
    res = mu.select_field_roi_mask([mask, mask], [img, img], preferred_idx=1)
    assert res['idx'] == 1 and res['selection'] == 'alias' and res['offset'] == (0, 0)
    assert res['n_different'] == 0
