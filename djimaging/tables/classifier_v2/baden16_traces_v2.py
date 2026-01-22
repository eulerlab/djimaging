"""
Preprocess traces to match Baden et al. 2016 dataset.
Mostly used to match traces for RGC classification.

V2 is updated to incorporate all processing steps in this table.
This is to ensure as few deviations from the original pipeline as possible.
"""

from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpolate

from djimaging.tables.core.averages import compute_upsampled_average
from djimaging.tables.core.preprocesstraces import process_trace
from djimaging.utils.dj_utils import get_primary_key
import datajoint as dj

from djimaging.utils.math_utils import normalize_amp_one
from djimaging.utils.snippet_utils import split_trace_by_reps
from djimaging.tables.response.movingbar.orientation_utils_v2 import compute_os_ds_idxs as compute_os_ds_idxs_v2


class Baden16TracesV2Template(dj.Computed):
    database = ""

    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'

    @property
    def definition(self):
        definition = """
        # Process traces to match Baden 2016 format, e.g. for RGC classifier.
        -> self.traces_table().proj(chirp_stim_name='stim_name')
        -> self.traces_table().proj(bar_stim_name='stim_name')
        ---
        preproc_chirp: blob  # preprocessed chirp trace (averaged, downsampled and normalized)
        preproc_bar:   blob  # preprocessed bar (time component in pref. dir., averaged and rolled)
        dir_component: blob  # projection of bar trace on direction component
        ds_index:      float # direction selectivity index as resulting vector length (absolute of projection on complex exponential)
        ds_pvalue:     float # p-value indicating the percentile of the vector length in null distribution
        pref_dir:      float # preferred direction
        os_index:      float # orientation selectivity index in analogy to ds_index
        os_pvalue:     float # analogous to ds_pvalue for orientation tuning
        pref_or:       float # preferred orientation
        """
        return definition

    @property
    def key_source(self):
        try:
            return ((self.traces_table & dict(stim_name=self._stim_name_chirp)).proj(chirp_stim_name='stim_name') *
                    (self.traces_table & dict(stim_name=self._stim_name_bar)).proj(bar_stim_name='stim_name'))
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

    def make(self, key):
        # Fetch chirp and preprocess
        chirp_stim_dict, chirp_ntrigger_rep = (self.stimulus_table() & dict(stim_name=self._stim_name_chirp)).fetch1(
            'stim_dict', 'ntrigger_rep')
        chirp_triggertimes = (self.presentation_table & dict(stim_name=self._stim_name_chirp) & key).fetch1(
            'triggertimes')
        chirp_trace, chirp_t0, chirp_dt = (self.traces_table & dict(stim_name=self._stim_name_chirp) & key).fetch1(
            'trace', 'trace_t0', 'trace_dt')

        qi, chirp_average = preprocess_chirp_v2(chirp_trace, chirp_t0, chirp_dt, chirp_triggertimes, chirp_ntrigger_rep)

        # Fetch bar and preprocess
        dir_order = (self.stimulus_table() & dict(stim_name=self._stim_name_bar) & key).fetch1('trial_info')
        bar_triggertimes = (self.presentation_table & dict(stim_name=self._stim_name_bar) & key).fetch1(
            'triggertimes')
        bar_trace, bar_t0, bar_dt = (self.traces_table & dict(stim_name=self._stim_name_bar) & key).fetch1(
            'trace', 'trace_t0', 'trace_dt')

        bar_qi, dsi, p_dsi, pref_dir, osi, p_osi, pref_or, time_component, dir_component = preprocess_bar_v2(
            bar_trace, bar_t0, bar_dt, bar_triggertimes, dir_order)

        key = key.copy()
        key['preproc_chirp'] = chirp_average
        key['preproc_bar'] = time_component
        key['dir_component'] = dir_component
        key['ds_index'] = dsi
        key['ds_pvalue'] = p_dsi
        key['pref_dir'] = pref_dir
        key['os_index'] = osi
        key['os_pvalue'] = p_osi
        key['pref_or'] = pref_or

        self.insert1(key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        preproc_chirp = (self & key).fetch1('preproc_chirp')
        preproc_bar = (self & key).fetch1('preproc_bar')

        fig, axs = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw=dict(width_ratios=(3, 1)))

        ax = axs[0]
        ax.plot(preproc_chirp)
        ax.set(xlabel='frames', ylabel='signal', title='chirp')

        ax = axs[1]
        ax.plot(preproc_bar)
        ax.set(xlabel='frames', ylabel='signal', title='bar')

        plt.show()


def preprocess_trace(trace: np.ndarray, t0: float, dt: float, triggertimes: np.ndarray, dt_baseline: float,
                     ntrigger_rep: int, poly_order: int, window_len_seconds: float, delay: float):
    # Lowpass filter
    pp_trace, smoothed_trace, pp_dt = process_trace(
        trace=trace, trace_t0=t0, trace_dt=dt,
        stim_start=triggertimes[0], poly_order=poly_order, window_len_seconds=window_len_seconds,
        subtract_baseline=True, standardize=1, non_negative=False,
        f_cutoff=None, fs_resample=None, baseline_max_dt=3)

    if not np.isclose(pp_dt, dt):
        raise ValueError(f"Processed trace dt {pp_dt} does not match original dt {dt}. "
                         f"Check if the processing parameters are correct.")

    # Split trace into snippets
    snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_reps(
        pp_trace, np.arange(len(pp_trace)) * dt + t0, triggertimes, ntrigger_rep,
        delay=delay, allow_drop_last=True, pad_trace=True)

    # Subtract baselines per snippet
    n_baseline = int(np.round(dt_baseline / dt))
    snippets = snippets - np.median(snippets[:n_baseline, :], axis=0)

    return snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag


def preprocess_chirp_v2(trace, t0, dt, triggertimes, ntrigger_rep=2, poly_order=3, window_len_seconds=60):
    snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = preprocess_trace(
        trace, t0, dt, triggertimes, dt_baseline=8 * 0.128,
        ntrigger_rep=ntrigger_rep, poly_order=poly_order, window_len_seconds=window_len_seconds, delay=0)

    qi = np.var(np.mean(snippets, axis=1)) / np.mean(np.var(snippets, axis=0))

    # Compute average and normalize
    average, average_times, _ = compute_upsampled_average(
        snippets, snippets_times, triggertimes_snippets, f_resample=7.8125)
    average = normalize_amp_one(average)

    # Resample to Baden frequency which was (in stimulus space) slightly different
    d1 = 0.128  # = 1 / 7.8125
    d2 = 1. / 7.5  # fs to "lose" 2 frames in the first chirp flicker

    ts1 = np.arange(0, 0 + 80) * d1
    ts2 = ts1[-1] + np.arange(1, 1 + 80) * d2
    ts3 = ts2[-1] + np.arange(1, 1 + 249 - len(ts1) - len(ts2)) * d1

    baden16like_times = np.concatenate([ts1, ts2, ts3])

    baden16like_average = interpolate.interp1d(
        average_times + 0.128, average, assume_sorted=True, bounds_error=False,
        fill_value=(average[0], average[-1]))(baden16like_times)

    # Normalize
    baden16like_average /= np.max(np.abs(baden16like_average))

    return qi, baden16like_average


def preprocess_bar_v2(trace, t0, dt, triggertimes, dir_order,
                      ntrigger_dir=1, poly_order=3, window_len_seconds=60, n_shuffles=1000):
    snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = preprocess_trace(
        trace, t0, dt, triggertimes, dt_baseline=5 * 0.128,
        ntrigger_rep=ntrigger_dir, poly_order=poly_order, window_len_seconds=window_len_seconds, delay=4 * 0.128)

    # Compute the preferred direction and projections
    dsi, p_dsi, null_dist_dsi, pref_dir, osi, p_osi, null_dist_osi, pref_or, \
        on_off, qi, time_component, dir_component, surrogate_v, dsi_s, avg_sorted_responses = \
        compute_os_ds_idxs_v2(snippets=snippets, dir_order=dir_order, dt=dt, n_shuffles=n_shuffles)

    # Normalize
    baden16like_average = time_component[:32]
    baden16like_average /= np.max(np.abs(baden16like_average))

    return qi, dsi, p_dsi, pref_dir, osi, p_osi, pref_or, baden16like_average, dir_component
