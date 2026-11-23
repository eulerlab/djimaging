import numpy as np
import pytest

from djimaging.utils.receptive_fields.temporal_rf_utils import (
    compute_half_amp_width, compute_main_peak_lag, compute_polarity_and_peak_idxs, compute_rel_weight_baseline,
    compute_trf_transience_index, get_main_and_pre_peak)

# rf_time convention: negative lags, 0 = time of the response
RF_TIME = np.linspace(-1., 0., 51)  # dt = 0.02 s


def _bump(t_peak, sigma, amp=1.):
    return amp * np.exp(-0.5 * ((RF_TIME - t_peak) / sigma) ** 2)


def _biphasic(main_amp=1., pre_amp=-0.5, t_main=-0.2, t_pre=-0.46, sigma=0.04):
    return _bump(t_main, sigma, main_amp) + _bump(t_pre, sigma, pre_amp)


# --- polarity and peaks ---

@pytest.mark.parametrize('sign', [1, -1])
def test_polarity_from_latest_peak(sign):
    trf = sign * _biphasic()
    polarity, peak_idxs = compute_polarity_and_peak_idxs(trf)
    assert polarity == sign
    np.testing.assert_array_equal(peak_idxs, [np.argmin(np.abs(RF_TIME + 0.46)), np.argmin(np.abs(RF_TIME + 0.2))])


def test_polarity_flat_trf_is_zero():
    polarity, peak_idxs = compute_polarity_and_peak_idxs(np.zeros(RF_TIME.size))
    assert polarity == 0
    assert peak_idxs.size == 0


def test_polarity_ignores_future_peaks():
    # Late negative peak is in the "future" and must be ignored
    trf = _bump(-0.5, 0.04, 1.) + _bump(-0.04, 0.02, -1.)
    polarity, _ = compute_polarity_and_peak_idxs(trf)
    assert polarity == -1
    polarity, peak_idxs = compute_polarity_and_peak_idxs(trf, rf_time=RF_TIME, max_dt_future=-0.1)
    assert polarity == 1
    assert np.all(RF_TIME[peak_idxs] <= -0.1)


def test_polarity_npeaks_max_keeps_latest_peaks():
    trf = _bump(-0.8, 0.03, 1.) + _bump(-0.5, 0.03, -1.) + _bump(-0.2, 0.03, 1.)
    _, peak_idxs_all = compute_polarity_and_peak_idxs(trf)
    assert peak_idxs_all.size == 3
    _, peak_idxs = compute_polarity_and_peak_idxs(trf, npeaks_max=2)
    np.testing.assert_array_equal(peak_idxs, peak_idxs_all[-2:])


def test_polarity_does_not_modify_input():
    trf = _biphasic()
    trf_copy = trf.copy()
    compute_polarity_and_peak_idxs(trf, rf_time=RF_TIME, max_dt_future=-0.3)
    np.testing.assert_array_equal(trf, trf_copy)


def test_main_peak_is_latest_of_two_largest_peaks():
    trf = _bump(-0.8, 0.03, 0.2) + _bump(-0.5, 0.03, -1.) + _bump(-0.2, 0.03, 0.7)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf, nstd=0.1)
    main_peak, t_main, main_idx, pre_peak, t_pre, pre_idx = get_main_and_pre_peak(RF_TIME, trf, peak_idxs)
    assert t_main == pytest.approx(-0.2)
    assert main_peak == pytest.approx(0.7, abs=0.01)
    assert t_pre == pytest.approx(-0.5)
    assert pre_peak == pytest.approx(-1., abs=0.01)


def test_main_and_pre_peak_single_peak_uses_opposite_extreme_before():
    trf = _bump(-0.2, 0.04, 1.) + _bump(-0.6, 0.1, -0.1)
    peak_idxs = np.array([np.argmax(trf)])
    main_peak, t_main, main_idx, pre_peak, t_pre, pre_idx = get_main_and_pre_peak(RF_TIME, trf, peak_idxs)
    assert t_main == pytest.approx(-0.2)
    assert pre_idx == np.argmin(trf[:main_idx])
    assert pre_peak < 0


def test_main_and_pre_peak_no_peaks():
    assert get_main_and_pre_peak(RF_TIME, _biphasic(), np.array([], dtype=int)) == (None,) * 6
    # All peaks in the future
    _, peak_idxs = compute_polarity_and_peak_idxs(_biphasic())
    assert get_main_and_pre_peak(RF_TIME, _biphasic(), peak_idxs, max_dt_future=-0.9) == (None,) * 6


# --- transience ---

@pytest.mark.parametrize('sign', [1, -1])
def test_transience_index_biphasic(sign):
    trf = sign * _biphasic(main_amp=1., pre_amp=-0.5)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf)
    ti = compute_trf_transience_index(RF_TIME, trf, peak_idxs)
    assert ti == pytest.approx(2 * 0.5 / 1.5, rel=0.01)


def test_transience_index_monophasic_and_dominant_pre_peak():
    trf = _bump(-0.2, 0.04)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf)
    assert compute_trf_transience_index(RF_TIME, trf, peak_idxs) == pytest.approx(0., abs=1e-6)

    # Pre peak larger than main peak
    trf = _biphasic(main_amp=0.6, pre_amp=-1.)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf)
    assert compute_trf_transience_index(RF_TIME, trf, peak_idxs) == 1.

    # Same-sign pre peak
    trf = _biphasic(main_amp=1., pre_amp=0.6)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf)
    assert compute_trf_transience_index(RF_TIME, trf, peak_idxs) == 0.


def test_transience_index_no_peaks():
    assert compute_trf_transience_index(RF_TIME, _biphasic(), np.array([], dtype=int)) is None


# --- main peak lag ---

@pytest.mark.parametrize('t_peak', [-0.2, -0.31, -0.555])
def test_main_peak_lag_positive_peak_subsample(t_peak):
    trf = _bump(t_peak, 0.05)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf)
    lag = compute_main_peak_lag(RF_TIME, trf, peak_idxs)
    assert lag == pytest.approx(-t_peak, abs=0.002)


def test_main_peak_lag_no_peaks():
    assert compute_main_peak_lag(RF_TIME, _bump(-0.3, 0.05), np.array([], dtype=int)) is None


@pytest.mark.xfail(strict=True, reason="BUG: compute_main_peak_lag uses argmax, wrong lag for negative main peaks")
def test_main_peak_lag_negative_peak():
    # TempRFProperties passes trf unflipped for split methods 'sd' and 'peak' (only SVD tRFs are flipped)
    trf = _bump(-0.3, 0.05, -1.)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf)
    lag = compute_main_peak_lag(RF_TIME, trf, peak_idxs)
    assert lag == pytest.approx(0.3, abs=0.002)


# --- half amplitude width ---

def test_half_amp_width_no_peaks():
    assert compute_half_amp_width(RF_TIME, _bump(-0.3, 0.05), np.array([], dtype=int)) is None


@pytest.mark.parametrize('sign', [1, -1])
def test_half_amp_width_is_fwhm_of_gaussian(sign):
    sigma = 0.05
    trf = sign * _bump(-0.4, sigma)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf)
    width = compute_half_amp_width(RF_TIME, trf, peak_idxs)
    assert width == pytest.approx(2 * np.sqrt(2 * np.log(2)) * sigma, rel=0.02)


# --- baseline weight ---

def test_rel_weight_baseline():
    rf_time = np.arange(-1., 0.01, 0.1)  # 11 samples
    trf = np.ones(rf_time.size)
    trf[:3] = -1.  # first 0.2 s (3 samples: -1, -0.9, -0.8)
    assert compute_rel_weight_baseline(rf_time, trf, dt_baseline=0.2) == pytest.approx(3 / 11)
    assert compute_rel_weight_baseline(rf_time, trf, dt_baseline=10.) == pytest.approx(1.)


def test_main_peak_lag_plot_smoke():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    trf = _bump(-0.3, 0.05)
    _, peak_idxs = compute_polarity_and_peak_idxs(trf)
    assert compute_main_peak_lag(RF_TIME, trf, peak_idxs, plot=True) == pytest.approx(0.3, abs=0.002)
    plt.close('all')
