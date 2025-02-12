"""
This table preprocesses the Contrast Step Light (CSL) response traces and extracts features from them.
It has its own detrending and normalization functions, so it gets raw traces as input.
It fits a sigmoid to the upper and lower bounds of the contrast steps.

Example usage:

from djimaging.tables.response import CslMetrics

@schema
class CslMetrics(response.CslMetricsTemplate):
    _stim_restriction = dict(stim_name='csl')

    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces

    _kind = 'naka_rushton'
    _w_zero_fit = True
    _dt_order = 3
    _dt_window = 60
    _peak_q = 98
    _contrast_levels = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    _dt_breaks = 3.
    _dt_baseline_a = 1.6
    _dt_baseline_b = 2.9
    _dt_window_plateau = 1.0

# Populate with plots:
CslMetrics().populate(make_kwargs=dict(plot=True))
"""

from abc import abstractmethod

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np

from djimaging.tables.core.preprocesstraces import process_trace
from djimaging.tables.response.csl.naka_rushton_utils import fit_naka_rushton
from djimaging.tables.response.csl.sigmoid_utils import fit_sigmoid
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.math_utils import normalize_zero_one
from djimaging.utils.snippet_utils import split_trace_by_reps


class CslMetricsTemplate(dj.Computed):
    database = ""
    _stim_restriction = dict(stim_name='csl')

    _fit_kind = 'naka_rushton'
    _w_zero_fit = 1
    _dt_order = 3
    _dt_window = 60
    _peak_q = 98
    _contrast_levels = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    _dt_breaks = 3.
    _dt_baseline_a = 1.6
    _dt_baseline_b = 2.9
    _dt_window_plateau = 1.0

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
        qidx_full: float  # Quality index for full trace (as in Baden et al 2016)
        qidx_contrast: float  # Quality index for contrast steps only (i.e. excluding half-steps)
        on_off_index: float  # Index indicating light preference (-1 Off, 1 On)
        contrast_sensitivity: float  # Relating step responses to contrast responses
        tonic_release_index: float  # Tonic release index as in Franke et al 2017, but for last contrast step
        plateau_index: float  # Plateau index (a - b) / (a + b), similar to Franke et al 2017
        contrast_aucs: blob  # Area under the curve for each contrast, incl. baseline at i=0 if _w_zero_fit=1
        fit_half_amp_y = NULL : float  # y at half amplitude of fit
        fit_half_amp_x = NULL : float  # x at half amplitude of fit
        fit_half_amp_slope = NULL : float  # Slope at half amplitude of fit
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
        trace_dt, trace_t0, trace = (self.traces_table & key).fetch1('trace_dt', 'trace_t0', 'trace')

        if len(trace) == 0:
            raise ValueError(f'Cannot compute CSL metrics for empty trace with key={key}')

        line_duration = (self.presentation_table().ScanInfo & key).fetch1('line_duration')
        scan_type, nypix, nzpix = (self.presentation_table & key).fetch1('scan_type', 'nypix', 'nzpix')
        n_lines = int(nzpix if scan_type == 'xz' else nypix)

        triggertimes = (self.presentation_table & key).fetch1('triggertimes')
        ntrigger_rep = (self.stimulus_table & key).fetch1('ntrigger_rep')

        fs_resample = 1 / line_duration

        d_csl = analyse_csl_response(
            trace, trace_t0, trace_dt, triggertimes, ntrigger_rep, fs_resample, plot=plot,
            w_zero_fit=self._w_zero_fit, fit_kind=self._fit_kind,
            dt_order=self._dt_order, dt_window=self._dt_window, peak_q=self._peak_q,
            contrast_levels=self._contrast_levels, dt_breaks=self._dt_breaks,
            dt_baseline_a=self._dt_baseline_a, dt_baseline_b=self._dt_baseline_b,
            dt_window_plateau=self._dt_window_plateau
        )

        # Downsample before saving
        fs = fs_resample / n_lines
        bc_snippets = d_csl['bc_snippets'][::n_lines, :]
        avg = d_csl['avg'][::n_lines]

        entry = dict(
            **key, average=avg, snippets=bc_snippets, fs=fs, fs_metrics=fs_resample,
            qidx_full=d_csl['qidx_full'], qidx_contrast=d_csl['qidx_contrast'],
            on_off_index=d_csl['on_off_index'], contrast_sensitivity=d_csl['contrast_sensitivity'],
            tonic_release_index=d_csl['tonic_release_index'], plateau_index=d_csl['plateau_index'],
            fit_half_amp_y=d_csl['fit'][0], fit_half_amp_x=d_csl['fit'][1], fit_half_amp_slope=d_csl['fit'][2],
            contrast_aucs=d_csl['contrast_aucs'], droppedlastrep_flag=d_csl['droppedlastrep_flag']
        )

        return entry

    def make(self, key, plot=False, verbose=False, DEBUG=False):
        if verbose:
            print(f'Populating {key}')

        trace = (self.traces_table & key).fetch1('trace')
        if len(trace) == 0:
            if verbose:
                print(f'Skipping CSL metrics for empty trace with key={key}')
            return

        entry = self._make_fetch_and_compute(key, plot=plot)
        if DEBUG:
            return
        self.insert1(entry)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        old_entry = (self & key).fetch1()
        new_entry = self._make_fetch_and_compute(key, plot=True)

        for k in ['qidx_full', 'qidx_contrast',
                  'on_off_index', 'contrast_sensitivity', 'tonic_release_index', 'plateau_index',
                  'fit_half_amp_y', 'fit_half_amp_x', 'fit_half_amp_slope']:
            if not np.isclose(old_entry[k], new_entry[k], atol=1e-3):
                raise ValueError(f'Plotted value {new_entry[k]} does not match database value {old_entry[k]}')


def analyse_csl_response(
        trace, trace_t0, trace_dt, triggertimes, ntrigger_rep, fs_resample, plot=False,
        w_zero_fit=1, fit_kind='naka_rushton',
        dt_order=3, dt_window=60, peak_q=98,
        contrast_levels=(0.10, 0.20, 0.40, 0.60, 0.80, 1.00),
        dt_breaks=3., dt_baseline_a=1.6, dt_baseline_b=2.9, dt_window_plateau=1.0):
    """Normalize contrast step light response. Detrend, resample, split, normalize, and extract features.
    Fit a sigmoid to the upper and lower bounds of the contrast steps.
    """

    contrast_levels = np.asarray(contrast_levels)

    if plot:
        fig, axs = plt.subplot_mosaic(
            [['A'], ['B'], ['C'], ['D'], ['E'], ['F'], ['G'], ['H']],
            figsize=(10, 15), height_ratios=[1] * 7 + [2])

    pp_trace, pp_smoothed_trace, pp_dt = process_trace(
        trace=trace, trace_t0=trace_t0, trace_dt=trace_dt,
        poly_order=dt_order, window_len_seconds=dt_window, fs_resample=fs_resample)

    pp_trace_times = np.arange(pp_trace.size) * pp_dt + trace_t0

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
    qidx_full = (np.var(np.mean(bc_snippets, axis=1)) /
                 np.mean(np.var(bc_snippets, axis=0)))

    idx_contrast_a, idx_contrast_b = idxs_cs_a[0], idxs_base_b[-1]

    qidx_contrast = (np.var(np.mean(bc_snippets[idx_contrast_a:idx_contrast_b, :], axis=1)) /
                     np.mean(np.var(bc_snippets[idx_contrast_a:idx_contrast_b, :], axis=0)))

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

    # Fit curve to responses
    contrast_aucs = np.array([np.mean(np.abs(avg[ia:ib])) for ia, ib in zip(idxs_cs_a, idxs_cs_b)])
    if w_zero_fit:
        zero_value = np.mean(np.abs(avg[idxs_baseline]))
        contrast_aucs = np.append(zero_value, contrast_aucs)
        contrast_levels = np.append(0, contrast_levels)

    if fit_kind == 'naka_rushton':
        fit = fit_naka_rushton(
            y_data=normalize_zero_one(contrast_aucs), x_data=contrast_levels, ax=None if not plot else axs['H'])
    elif fit_kind == 'sigmoid':
        fit = fit_sigmoid(
            y_data=normalize_zero_one(contrast_aucs), x_data=contrast_levels, ax=None if not plot else axs['H'])
    else:
        raise ValueError(f'Unknown fit_kind={fit_kind}')

    if plot:
        ax = axs['A']
        ax.set_title('Trace')
        ax.plot(np.arange(trace.size) * trace_dt + trace_t0, trace)
        ax.vlines(triggertimes, np.min(trace), np.max(trace), color='r')
        ax.axhline(np.median(trace), c='dimgray', ls='--')

        ax = axs['B']
        ax.set_title('Detrended trace')
        ax.plot(pp_trace_times, pp_trace)
        ax.vlines(triggertimes, np.min(pp_trace), np.max(pp_trace), color='r')
        ax.axhline(0, c='dimgray', ls='--')

        ax = axs['C']
        ax.set_title(f'Snippets & Local baselines')
        for i, (snip, base) in enumerate(zip(snippets.T, baselines.T)):
            ax.plot(rel_time, snip, alpha=0.5, c=f'C{i}')
            ax.plot(rel_time, base, alpha=1, c=f'C{i}')
        ax.vlines(rel_triggertimes, np.min(snippets), np.max(snippets), color='r')
        for ia, ib in zip(idxs_base_a, idxs_base_b):
            ax.plot([rel_time[ia], rel_time[ib]], [np.mean(baselines)] * 2, '|-', color='magenta')
        for ia, ib in zip(idxs_cs_a, idxs_cs_b):
            ax.plot([rel_time[ia], rel_time[ib]], [np.mean(baselines) - 0.2] * 2, '|-', color='k')

        ax = axs['D']
        ax.set_title(f'Normalized baseline-corrected snippets & Average: '
                     f'qidx_full={qidx_full:.2f} & qidx_contrast={qidx_contrast:.2f}')
        ax.plot(rel_time, bc_snippets, alpha=0.5)
        ax.plot(rel_time, avg, c='k')
        ax.vlines(rel_triggertimes, np.min(bc_snippets), np.max(bc_snippets), color='r')
        # ax.plot([rel_time[idx_contrast_a], rel_time[idx_contrast_b]], [np.mean(baselines) - 0.2] * 2, '|-', color='k')

        ax = axs['E']
        ax.set_title(f'TRi={tonic_release_index:.2f}; RPi={plateau_index:.2f}')
        ax.plot(rel_time, avg, c='k')

        # TRi - Tonic release index
        ax.plot([rel_time[idxs_cs_a[-1]], rel_time[idxs_cs_b[-1]]], [np.min(avg)] * 2, 'k|-')
        ax.fill_between(rel_time[idxs_cs_a[-1]:idxs_cs_b[-1]], np.clip(avg_100, None, 0),
                        color='r', alpha=0.5, zorder=10)
        ax.fill_between(rel_time[idxs_cs_a[-1]:idxs_cs_b[-1]], np.clip(avg_100, 0, None),
                        color='g', alpha=0.5, zorder=10)
        ax.axhline(0, ls=':', color='gray')

        # RPi- Response plateau index
        ax.plot([rel_time[idx_pt1_a], rel_time[idx_pt1_b]], [pt1_amp] * 2, 'X-', c='k')
        ax.plot([rel_time[idx_pt2_a], rel_time[idx_pt2_b]], [pt2_amp] * 2, 'X-', c='gray')
        ax.axhline(0, ls=':', color='gray')

        # On off index and contrast sensitivity
        ax = axs['F']
        ax.set_title(f'OnOffI={on_off_index:.2f}; CS={contrast_sensitivity:.2f}')
        ax.plot([rel_time[idx_onstep_a], rel_time[idx_onstep_b]], [amp_on] * 2, 'rX-', label='On')
        ax.plot([rel_time[idx_offstep_a], rel_time[idx_offstep_b]], [amp_off] * 2, 'bX-', label='Off')
        ax.plot([rel_time[idxs_cs_a[-1]], rel_time[idxs_cs_b[-1]]], [amp_100] * 2, 'gX-', label='100%')
        ax.plot(rel_time, avg, c='k')
        ax.legend(loc='upper center')
        ax.axhline(0, ls=':', color='gray')

        # Fit
        ax = axs['G']
        ax.set_title('Response metric for contrast steps')
        ax.plot(rel_time, avg, c='k', lw=0.8, alpha=0.8)

        if w_zero_fit:
            for i, (ia, ib) in enumerate(zip(idxs_base_a, idxs_base_b)):
                ax.fill_between(rel_time[ia:ib], np.zeros_like(avg[ia:ib]), avg[ia:ib], color='C0', lw=0)

        for i, (ia, ib) in enumerate(zip(idxs_cs_a, idxs_cs_b)):
            ax.fill_between(rel_time[ia:ib], np.zeros_like(avg[ia:ib]), avg[ia:ib], color=f'C{i + 1}', lw=0)

        ax.axhline(0, ls=':', color='gray')
        ax2 = ax.twinx()
        ax2.bar(x=rel_time[(idxs_cs_a + idxs_cs_b) // 2], height=contrast_aucs[-len(idxs_cs_a):], color='k', alpha=0.5)
        vabsmax = np.max(np.abs(contrast_aucs[-len(idxs_cs_a):]))
        ax2.set_ylim(-vabsmax * 1.1, vabsmax * 1.1)
        ax2.axhline(0, ls='--', color='gray')

        axs['H'].set_title('Fit')

        plt.tight_layout()
        plt.show()

    result_dict = dict(
        bc_snippets=bc_snippets, avg=avg, qidx_full=qidx_full, qidx_contrast=qidx_contrast,
        tonic_release_index=tonic_release_index,
        plateau_index=plateau_index, on_off_index=on_off_index, contrast_sensitivity=contrast_sensitivity,
        contrast_aucs=contrast_aucs, fit=fit, droppedlastrep_flag=droppedlastrep_flag
    )

    return result_dict
