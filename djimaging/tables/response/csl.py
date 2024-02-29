"""
This table preprocesses the Contrast Step Light (CSL) response traces and extracts features from them.
It has its own detrending and normalization functions, so it gets raw traces as input.
It fits a sigmoid to the upper and lower bounds of the contrast steps.

Example usage:

from djimaging.tables.response import CslMetrics

@schema
class CslMetrics(response.CslMetricsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces

# Populate with plots:
CslMetrics().populate(make_kwargs=dict(plot=True))
"""

from abc import abstractmethod

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.scanm_utils import split_trace_by_reps
from djimaging.tables.core.preprocesstraces import process_trace
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datajoint as dj


class CslMetricsTemplate(dj.Computed):
    database = ""
    _stim_restriction = dict(stim_name='csl')

    @property
    def definition(self):
        definition = '''
        # Normalized contrast step light response
        -> self.traces_table
        ---
        average: longblob  # Average of the normalized snippets (time, )
        snippets: longblob  # Baseline corrected snippets (times x repetitions)
        fs: float  # Sampling frequency in which average and snippets are stored
        fs_metrics: float  # Sampling frequency used to compute metrics
        qidx: float  # Quality index as in Baden et al 2016
        on_off_index: float  # Index indicating light preference (-1 Off, 1 On)
        contrast_sensitivity: float  # Relating step responses to contrast responses
        tonic_release_index: float  # Tonic release index as in Franke et al 2017, but for last contrast step
        plateau_index: float  # Plateau index (a - b) / (a + b), similar to Franke et al 2017
        ub_half_amp: float  # Upper bound half amplitude
        ub_half_amp_x: float  # Upper bound half amplitude x
        ub_slope_half_amp: float  # Upper bound slope at half amplitude
        lb_half_amp: float  # Lower bound half amplitude
        lb_half_amp_x: float  # Lower bound half amplitude x
        lb_slope_half_amp: float  # Lower bound slope at half amplitude
        droppedlastrep_flag: tinyint unsigned  # Was the last repetition incomplete and therefore dropped?
        '''
        return definition

    @property
    def key_source(self):
        try:
            return self.traces_table.proj() & self._stim_restriction
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def _make_fetch_and_compute(self, key, plot=False):
        trace_times, trace = (self.traces_table & key).fetch1('trace_times', 'trace')

        if len(trace) == 0:
            raise ValueError(f'Cannot compute CSL metrics for empty trace with key={key}')

        scan_type = (self.presentation_table & key).fetch1('scan_type')
        line_duration, nypix, nzpix = (self.presentation_table.ScanInfo() & key).fetch1(
            'line_duration', 'user_dypix', 'user_dzpix')
        n_lines = int(nzpix if scan_type == 'xz' else nypix)

        triggertimes = (self.presentation_table & key).fetch1('triggertimes')
        ntrigger_rep = (self.stimulus_table & key).fetch1('ntrigger_rep')

        fs_resample = 1 / line_duration

        d_csl = analyse_csl_response(trace_times, trace, triggertimes, ntrigger_rep, fs_resample, plot=plot)

        # Downsample before saving
        fs = fs_resample / n_lines
        bc_snippets = d_csl['bc_snippets'][::n_lines, :]
        avg = d_csl['avg'][::n_lines]

        entry = dict(
            key, average=avg, snippets=bc_snippets, fs=fs, fs_metrics=fs_resample, qidx=d_csl['qidx'],
            on_off_index=d_csl['on_off_index'], contrast_sensitivity=d_csl['contrast_sensitivity'],
            tonic_release_index=d_csl['tonic_release_index'], plateau_index=d_csl['plateau_index'],
            ub_half_amp=d_csl['ub_fit'][0], ub_half_amp_x=d_csl['ub_fit'][1], ub_slope_half_amp=d_csl['ub_fit'][2],
            lb_half_amp=d_csl['lb_fit'][0], lb_half_amp_x=d_csl['lb_fit'][1], lb_slope_half_amp=d_csl['lb_fit'][2],
            droppedlastrep_flag=d_csl['droppedlastrep_flag']
        )

        return entry

    def make(self, key, plot=False, verbose=False):
        if verbose:
            print(f'Populating {key}')

        trace = (self.traces_table & key).fetch1('trace')
        if len(trace) == 0:
            if verbose:
                print(f'Skipping CSL metrics for empty trace with key={key}')
            return

        entry = self._make_fetch_and_compute(key, plot=plot)
        self.insert1(entry)

    def plot1(self, key):
        key = get_primary_key(table=self, key=key)
        old_entry = (self & key).fetch1()
        new_entry = self._make_fetch_and_compute(key, plot=True)

        if np.all([old_entry[k] == new_entry[k] for k in ['tonic_release_index', 'plateau_index', 'qidx']]):
            raise ValueError('Plotted values are not identical to stored values.')


def sigmoid(x, x0, k, a):
    """Define sigmoid function"""
    y = a / (1 + np.exp(-k * (x - x0)))
    return y


def init_sigmoid_params(x_data, y_data):
    """Initialize sigmoid parameters"""
    p0 = [x_data[np.argmax(y_data > (y_data.max() / 2))], 10., np.percentile(y_data, 90)]
    bounds = [(x_data[0], 1e-9, np.min(y_data)), (x_data[-1], np.inf, 1.2 * np.max(y_data))]
    return p0, bounds


def fit_sigmoid_with_retry(x_data, y_data, max_tries=3):
    p00, bounds = init_sigmoid_params(x_data, y_data)

    for i in range(max_tries):
        try:
            p0 = p00 if i == 0 else [np.random.uniform(a, np.minimum(b, 100)) for a, b in np.array(bounds).T]
            popt = curve_fit(sigmoid, x_data, y_data, p0=p0, bounds=bounds)[0]
            return popt
        except RuntimeError:
            pass
    return p00


def fit_sigmoid(y_data, x_data=None, sign=1., ax=None):
    """Fit sigmoid function to data, and estimate half amp"""
    np.random.seed(42)

    if x_data is None:
        x_data = np.arange(y_data.size)

    # Make sure y_data is positive before fitting, and invert at the end again if necessary
    y_data = sign * y_data

    # Fit sigmoid curve to the data
    popt = fit_sigmoid_with_retry(x_data, y_data, max_tries=3)
    x0_fit, k_fit, a_fit = popt

    if (x0_fit > x_data.max()) | (x0_fit < x_data.min()):
        x0_fit = 0

    # Calculate half amplitude x value
    half_amplitude = sign * a_fit / 2
    half_amplitude_x = x0_fit
    slope_at_half_amplitude = sign * k_fit * half_amplitude * (1 - half_amplitude / a_fit)

    if ax is not None:
        x_data_us = np.linspace(x_data[0], x_data[-1], x_data.size * 20)
        ax.scatter(x_data, sign * y_data, label='Data')
        ax.plot(x_data_us, sign * sigmoid(x_data_us, *popt), 'r-',
                label='Fit: x0=%5.3f, k=%5.3f, A=%5.3f' % tuple(popt))
        ax.plot(half_amplitude_x, half_amplitude, 'gD', linestyle='--',
                label=f'x(Half Amplitude)={half_amplitude_x:.2f}')
        dt = np.min(np.diff(x_data))
        ax.plot([half_amplitude_x - 0.5 * dt, half_amplitude_x + 0.5 * dt],
                [half_amplitude - 0.5 * dt * slope_at_half_amplitude,
                 half_amplitude + 0.5 * dt * slope_at_half_amplitude],
                color='k', label=f'slope={slope_at_half_amplitude:.2f}')
        ax.legend(fontsize=6)

    return half_amplitude, half_amplitude_x, slope_at_half_amplitude


def analyse_csl_response(trace_times, trace, triggertimes, ntrigger_rep, fs_resample, plot=False,
                         w_zero_fit=1, dt_order=3, dt_window=60, peak_q=98,
                         contrast_levels=(0.10, 0.20, 0.40, 0.60, 0.80, 1.00),
                         dt_breaks=3., dt_baseline_a=1.6, dt_baseline_b=2.9, dt_window_plateau=1.0):
    """Normalize contrast step light response. Detrend, resample, split, normalize, and extract features.
    Fit a sigmoid to the upper and lower bounds of the contrast steps.
    """

    contrast_levels = np.asarray(contrast_levels)

    if plot:
        fig, axs = plt.subplot_mosaic(
            [['A'] * 2, ['B'] * 2, ['C'] * 2, ['D'] * 2, ['E'] * 2, ['F'] * 2, ['G'] * 2, ['H', 'I']],
            figsize=(10, 10), height_ratios=[1] * 7 + [2])
    pp_trace_times, pp_trace, pp_smoothed_trace = process_trace(
        trace_times, trace, poly_order=dt_order, window_len_seconds=dt_window, fs_resample=fs_resample)

    # Compute snippets
    snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_reps(
        pp_trace, pp_trace_times, triggertimes, ntrigger_rep, allow_drop_last=True, pad_trace=False)

    t0s = snippets_times[0, :]
    rel_time = snippets_times[:, 0] - t0s[0]
    rel_triggertimes = np.median(triggertimes_snippets - t0s, axis=1)

    # Get indexes of contrast steps
    idxs_cs_a = np.array([int(np.floor((tt + dt_breaks) * fs_resample).astype(int)) for tt in rel_triggertimes[:-2]])
    idxs_cs_b = np.array([int(np.floor(tt * fs_resample).astype(int)) for tt in rel_triggertimes[1:-1]])

    # Subtract baseline
    idxs_base_a = np.floor((rel_triggertimes[1:-1] + dt_baseline_a) * fs_resample).astype(int)
    idxs_base_b = np.floor((rel_triggertimes[1:-1] + dt_baseline_b) * fs_resample).astype(int)

    idxs_baseline = np.concatenate([np.arange(a, b) for a, b in zip(idxs_base_a, idxs_base_b)])

    baselines = np.zeros_like(snippets)
    for i, snippet_i in enumerate(snippets.T):
        m, y0 = np.polyfit(rel_time[idxs_baseline], snippet_i[idxs_baseline], deg=1)
        baselines[:, i] = rel_time * m + y0

    bc_snippets = snippets - baselines
    avg = np.mean(bc_snippets, axis=1)

    # Normalize scale
    scale = np.max(np.abs(avg[idxs_cs_a[0]:]))
    avg /= scale
    bc_snippets /= scale

    # Quality index - See Baden et al 2016
    qidx = np.var(np.mean(bc_snippets, axis=1)) / np.mean(np.var(bc_snippets, axis=0))

    # Tonic release index
    avg_100 = avg[idxs_cs_a[-1]:idxs_cs_b[-1]]
    tonic_release_index = np.sum(np.abs(avg_100[avg_100 < 0])) / np.sum(np.abs(avg_100))

    # Plateau index
    idx_pt1_a = idxs_cs_a[-1]
    idx_pt1_b = idxs_cs_a[-1] + int(dt_window_plateau * fs_resample)

    idx_pt2_a = idxs_cs_b[-1] - int(dt_window_plateau * fs_resample)
    idx_pt2_b = idxs_cs_b[-1]

    pt1_amp = np.maximum(0, np.percentile(avg[idx_pt1_a:idx_pt1_b], q=peak_q))
    pt2_amp = np.maximum(0, np.percentile(avg[idx_pt2_a:idx_pt2_b], q=peak_q))

    plateau_index = (pt1_amp - pt2_amp) / np.maximum(pt1_amp + pt2_amp, 1e-9)

    # Half-step responses for On and Off, compare to peak during the highest contrast
    idx_onstep_a = np.maximum(0, int(np.floor(rel_triggertimes[0] * fs_resample)))
    idx_onstep_b = idx_onstep_a + int(dt_window_plateau * fs_resample)

    amp_on = np.maximum(0, np.percentile(avg[idx_onstep_a:idx_onstep_b], q=peak_q))

    idx_offstep_a = int(np.floor(rel_triggertimes[-1] * fs_resample))
    idx_offstep_b = idx_offstep_a + int(dt_window_plateau * fs_resample)
    amp_off = np.maximum(0, np.percentile(avg[idx_offstep_a:idx_offstep_b], q=peak_q))

    on_off_index = (amp_on - amp_off) / np.maximum(amp_on + amp_off, 1e-9)

    # Compare step responses to contrast responses
    amp_step = np.maximum(amp_on, amp_off)
    amp_100 = np.maximum(0, np.percentile(avg[idxs_cs_a[-1]:idxs_cs_b[-1]], q=peak_q))
    contrast_sensitivity = amp_100 / np.maximum(amp_100 + amp_step, 1e-9)

    # Envelope sigmoid fit
    cs_bounds = np.array(
        [(np.percentile(avg[ia:ib], q=100 - peak_q), np.percentile(avg[ia:ib], q=peak_q))
         for ia, ib in zip(idxs_cs_a, idxs_cs_b)])

    x_data = np.append(np.zeros(w_zero_fit), contrast_levels)
    ub_fit = fit_sigmoid(y_data=np.append(np.zeros(w_zero_fit), cs_bounds[:, 1]), x_data=x_data,
                         ax=None if not plot else axs['H'])
    lb_fit = fit_sigmoid(y_data=np.append(np.zeros(w_zero_fit), cs_bounds[:, 0]), x_data=x_data,
                         ax=None if not plot else axs['I'], sign=-1)

    if plot:
        ax = axs['A']
        ax.set_title('Trace')
        ax.plot(trace_times, trace)
        ax.vlines(triggertimes, np.min(trace), np.max(trace), color='r')
        ax.axhline(np.median(trace), c='dimgray', ls='--')

        ax = axs['B']
        ax.set_title('Detrended trace')
        ax.plot(pp_trace_times, pp_trace)
        ax.vlines(triggertimes, np.min(pp_trace), np.max(pp_trace), color='r')
        ax.axhline(0, c='dimgray', ls='--')

        ax = axs['C']
        ax.set_title(f'Snippets & Local baselines; qidx={qidx:.2f}')
        for i, (snip, base) in enumerate(zip(snippets.T, baselines.T)):
            ax.plot(rel_time, snip, alpha=0.5, c=f'C{i}')
            ax.plot(rel_time, base, alpha=1, c=f'C{i}')
        ax.vlines(rel_triggertimes, np.min(snippets), np.max(snippets), color='r')
        for ia, ib in zip(idxs_base_a, idxs_base_b):
            ax.plot([rel_time[ia], rel_time[ib]], [np.mean(baselines)] * 2, 'k|-')

        ax = axs['D']
        ax.set_title('Normalized Baseline-corrected snippets & Average')
        ax.plot(rel_time, bc_snippets, alpha=0.5)
        ax.plot(rel_time, avg, c='k')
        ax.vlines(rel_triggertimes, np.min(bc_snippets), np.max(bc_snippets), color='r')

        ax = axs['E']
        ax.set_title(f'TRi={tonic_release_index:.2f}; RPi={plateau_index:.2f}')
        ax.plot(rel_time, avg, c='k')

        # TRi - Tonic release index
        ax.plot([rel_time[idxs_cs_a[-1]], rel_time[idxs_cs_b[-1]]], [np.min(avg)] * 2, 'k|-')
        ax.fill_between(rel_time[idxs_cs_a[-1]:idxs_cs_b[-1]], np.clip(avg_100, None, 0),
                        color='r', alpha=0.5, zorder=10)
        ax.fill_between(rel_time[idxs_cs_a[-1]:idxs_cs_b[-1]], np.clip(avg_100, 0, None),
                        color='g', alpha=0.5, zorder=10)

        # RPi- Response plateau index
        ax.plot([rel_time[idx_pt1_a], rel_time[idx_pt1_b]], [pt1_amp] * 2, 'X-', c='k')
        ax.plot([rel_time[idx_pt2_a], rel_time[idx_pt2_b]], [pt2_amp] * 2, 'X-', c='gray')

        # On off index and contrast sensitivity
        ax = axs['F']
        ax.set_title(f'OnOffI={on_off_index:.2f}; CS={contrast_sensitivity:.2f}')
        ax.plot([rel_time[idx_onstep_a], rel_time[idx_onstep_b]], [amp_on] * 2, 'rX-', label='On')
        ax.plot([rel_time[idx_offstep_a], rel_time[idx_offstep_b]], [amp_off] * 2, 'bX-', label='Off')
        ax.plot([rel_time[idxs_cs_a[-1]], rel_time[idxs_cs_b[-1]]], [amp_100] * 2, 'gX-', label='100%')
        ax.plot(rel_time, avg, c='k')
        ax.legend(loc='upper center')

        ax = axs['G']
        ax.set_title('Upper and lower bounds for contrast steps')
        ax.plot(rel_time, avg, c='k')
        ax.plot(rel_time[(idxs_cs_a + idxs_cs_b) // 2], cs_bounds[:, 0], 'rX')
        ax.plot(rel_time[(idxs_cs_a + idxs_cs_b) // 2], cs_bounds[:, 1], 'gX')
        for ia, ib, (lb, ub) in zip(idxs_cs_a, idxs_cs_b, cs_bounds):
            ax.fill_between([rel_time[ia], rel_time[ib]], [lb] * 2, [ub] * 2, color='c')

        axs['H'].set_title('Sigmoid fit UB')
        axs['I'].set_title('Sigmoid fit LB')

        plt.tight_layout()
        plt.show()

    result_dict = dict(
        bc_snippets=bc_snippets, avg=avg, qidx=qidx, tonic_release_index=tonic_release_index,
        plateau_index=plateau_index, on_off_index=on_off_index, contrast_sensitivity=contrast_sensitivity,
        ub_fit=ub_fit, lb_fit=lb_fit, droppedlastrep_flag=droppedlastrep_flag
    )

    return result_dict
