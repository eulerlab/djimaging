"""Tests for the color flicker center-surround RF metrics and the SR101 index, computed without a database."""
import warnings

import matplotlib

matplotlib.use('Agg')

import datajoint as dj  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from djimaging.tables.color_flicker.center_surround import CenterSurroundTemplate  # noqa: E402
from djimaging.tables.misc.sr_index import compute_sr_idxs  # noqa: E402
from tests.tables._fake_dj import FakeRel, make_fake  # noqa: E402

RF_TIME = np.linspace(-0.6, 0., 61)

XFAIL_HALF_AMP_WIDTH = pytest.mark.xfail(
    strict=True, raises=TypeError,
    reason="BUG: compute_half_amp_width calls float() on the 1-element OptimizeResult.x "
           "(TypeError with numpy>=2.5): CenterSurround.make crashes")


def _gauss(t_peak, sigma=0.03, amp=1.):
    return amp * np.exp(-(RF_TIME - t_peak) ** 2 / (2 * sigma ** 2))


def _cs_table(rf, rf_time_row=None, peak_nstd=1., npeaks_max=0):
    rf_time_row = dict(rf_time=RF_TIME) if rf_time_row is None else rf_time_row
    return make_fake(CenterSurroundTemplate, ['make', 'fetch1_rf_time'], _max_dt_future=np.inf,
                     color_rf_table=FakeRel(dict(rf=rf)), color_rf_time_table=FakeRel(rf_time_row),
                     cs_params_table=FakeRel(dict(peak_nstd=peak_nstd, npeaks_max=npeaks_max)))


def _make(table):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        table.make(dict(roi_id=1, cs_params_id=1))
    return table.inserted[0]


@XFAIL_HALF_AMP_WIDTH
def test_center_surround_on_center_off_surround():
    sigma = 0.03
    center = _gauss(-0.1, sigma)
    surround = _gauss(-0.15, sigma, amp=-0.5)
    entry = _make(_cs_table(np.stack([center, surround], axis=1)[None]))

    assert entry['c_polarity_idx'] == 1
    assert entry['s_polarity_idx'] == -1
    assert entry['c_main_peak_lag'] == pytest.approx(0.1, abs=2e-3)
    assert entry['s_main_peak_lag'] == pytest.approx(0.15, abs=2e-3)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert entry['c_half_amp_width'] == pytest.approx(fwhm, rel=0.05)
    assert entry['s_half_amp_width'] == pytest.approx(fwhm, rel=0.05)
    assert entry['c_transience_idx'] == pytest.approx(0.)  # monophasic


@XFAIL_HALF_AMP_WIDTH
def test_center_surround_biphasic_transience():
    # Biphasic kernel: main On peak (closest to t=0) preceded by an Off lobe of half the amplitude
    kernel = _gauss(-0.1) - 0.5 * _gauss(-0.25)
    entry = _make(_cs_table(np.stack([kernel, kernel], axis=1)))
    assert entry['c_polarity_idx'] == 1
    # 2 * |pre| / (|main| + |pre|)
    assert entry['c_transience_idx'] == pytest.approx(2 * 0.5 / 1.5, abs=0.02)


def test_center_surround_flat_kernel_defaults_to_minus_one():
    flat = np.zeros_like(RF_TIME)
    entry = _make(_cs_table(np.stack([flat, flat], axis=1)))
    for prefix in ['c', 's']:
        assert entry[f'{prefix}_polarity_idx'] == 0
        for name in ['transience_idx', 'half_amp_width', 'main_peak_lag']:
            assert entry[f'{prefix}_{name}'] == -1.


def test_center_surround_rf_time_from_model_dict():
    table = _cs_table(np.stack([_gauss(-0.1), _gauss(-0.1)], axis=1),
                      rf_time_row=dict(model_dict=dict(rf_time=RF_TIME)))
    np.testing.assert_array_equal(table.fetch1_rf_time(dict(roi_id=1)), RF_TIME)


def test_center_surround_rejects_non_2d_rf():
    table = _cs_table(np.zeros((RF_TIME.size, 2, 3)))
    with pytest.raises(ValueError, match='2d'):
        table.make(dict(roi_id=1))


def test_fake_fetch1_missing_attribute_raises_datajoint_error():
    with pytest.raises(dj.DataJointError):
        FakeRel(dict(a=1)).fetch1('b')


# ---------------------------------------------------------------- SR index

def _sr_stack(nx=32, ny=20, nt=10, npixartifact=3, bg=1., artifact=11., roi_values=(6., 3.5)):
    stack = np.full((nx, ny, nt), bg)
    stack[:npixartifact] = artifact
    roi_mask = np.zeros((nx, ny), dtype=int)
    roi_mask[10:14, 5:9] = 1
    roi_mask[20:23, 10:15] = 2
    stack[roi_mask == 1] = roi_values[0]
    stack[roi_mask == 2] = roi_values[1]
    return stack, roi_mask


def test_sr_index_known_values():
    stack, roi_mask = _sr_stack()
    sr_idxs, roi_avgs, roi_means, lb, ub = compute_sr_idxs(stack, np.array([1, 2]), roi_mask, npixartifact=3)
    assert lb == pytest.approx(1.)
    assert ub == pytest.approx(11.)
    np.testing.assert_allclose(roi_means, [6., 3.5])
    np.testing.assert_allclose(sr_idxs, [0.5, 0.25])
    assert roi_avgs[0].size == 16


def test_sr_index_invariant_to_affine_intensity_change():
    stack, roi_mask = _sr_stack()
    sr1 = compute_sr_idxs(stack, np.array([1, 2]), roi_mask, npixartifact=3)[0]
    sr2 = compute_sr_idxs(4 * stack + 9, np.array([1, 2]), roi_mask, npixartifact=3)[0]
    np.testing.assert_allclose(sr1, sr2)


def test_sr_index_igor_format_mask():
    stack, roi_mask = _sr_stack()
    igor_mask = np.where(roi_mask > 0, -roi_mask, 1)
    sr_idxs = compute_sr_idxs(stack, np.array([1, 2]), igor_mask, npixartifact=3)[0]
    np.testing.assert_allclose(sr_idxs, [0.5, 0.25])


def test_sr_index_roi_without_pixels_is_nan():
    stack, roi_mask = _sr_stack()
    with pytest.warns(UserWarning, match='no pixels'):
        sr_idxs = compute_sr_idxs(stack, np.array([1, 3]), roi_mask, npixartifact=3)[0]
    assert sr_idxs[0] == pytest.approx(0.5)
    assert np.isnan(sr_idxs[1])


def _sr_table(monkeypatch, stack, roi_mask, triggertimes, scan_frequency):
    from djimaging.tables.misc import sr_index
    from djimaging.tables.misc.sr_index import SrIndexTemplate

    monkeypatch.setattr(sr_index.read_utils, 'load_stacks',
                        lambda filepath, from_raw_data, ch_names: ({'wDataCh1': stack, 'wDataCh0': stack}, dict()))
    presentation = FakeRel(dict(triggertimes=triggertimes, h5_path='rec.h5', npixartifact=3), filepath='h5_path',
                           ScanInfo=FakeRel(dict(scan_frequency=scan_frequency)))
    return make_fake(SrIndexTemplate, ['make', 'fetch_and_compute'], presentation_table=presentation,
                     roimask_table=FakeRel(dict(roi_mask=roi_mask)), stimulus_table=FakeRel(dict(ntrigger_rep=1)),
                     raw_params_table=FakeRel(dict(from_raw_data=False)),
                     roi_table=FakeRel(rows=[dict(roi_id=1), dict(roi_id=2)]))


def _sr_stack_with_stimulus_window(nt=400, idx_a=200, idx_b=300):
    stack, roi_mask = _sr_stack(nt=nt)
    # Outside the stimulus window the ROIs are at background and there is no light artifact
    outside = np.ones(nt, dtype=bool)
    outside[idx_a:idx_b] = False
    stack[roi_mask > 0, :] = np.where(outside, 1., stack[roi_mask > 0, :])
    stack[:3, :, outside] = 1.
    return stack, roi_mask


def test_sr_index_make_uses_stimulus_window(monkeypatch):
    stack, roi_mask = _sr_stack_with_stimulus_window()
    table = _sr_table(monkeypatch, stack, roi_mask, triggertimes=np.array([20., 30.]), scan_frequency=10.)
    table.make(dict(field='f1', stim_name='gChirp'))
    np.testing.assert_allclose([r['sr_idx'] for r in table.inserted], [0.5, 0.25])


class _ScalarScanInfo(FakeRel):
    def to_arrays(self, *attrs):
        return np.float64(self.rows[0][attrs[0]])


def test_sr_index_make_with_scalar_scan_frequency(monkeypatch):
    # Same as above, but scan_frequency is returned as a scalar: shows the stimulus window logic itself is right
    stack, roi_mask = _sr_stack_with_stimulus_window()
    table = _sr_table(monkeypatch, stack, roi_mask, triggertimes=np.array([20., 30.]), scan_frequency=10.)
    table.presentation_table.ScanInfo = _ScalarScanInfo(dict(scan_frequency=10.))
    table.make(dict(field='f1', stim_name='gChirp'))
    assert [r['roi_id'] for r in table.inserted] == [1, 2]
    np.testing.assert_allclose([r['sr_idx'] for r in table.inserted], [0.5, 0.25])
