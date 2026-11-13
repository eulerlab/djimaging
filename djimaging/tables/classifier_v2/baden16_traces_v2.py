"""
Preprocess traces to match Baden et al. 2016 dataset.
Mostly used to match traces for RGC classification.

V2 is updated to incorporate all processing steps of the chirp in this table, to ensure as few deviations from the
original pipeline as possible. The moving bar features (time component and DS p-value) are read from the OsDsIndexes
table, which follows the Baden et al. 2016 implementation. The classifier was trained on these features, so OsDsIndexes
must use the Baden et al. 2016 per-trial snippet baseline (Stimulus.snippet_base_dt = 5 * 0.128 s), 1000 shuffles and
snippets at 7.8125 Hz. Under these conditions the features are identical to the former computation inside this table.

Example usage:

@schema
class Baden16TracesV2(classifier_v2.Baden16TracesV2Template):
    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'

    traces_table = Traces
    presentation_table = Presentation
    stimulus_table = Stimulus
    osds_table = OsDsIndexes
"""

from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from djimaging.tables.core.averages import compute_upsampled_average
from djimaging.tables.core.preprocesstraces import process_trace
from djimaging.tables.core.stimulus import MOVINGBAR_SNIPPET_BASE_DT
from djimaging.utils.dj_storage import load_array
from djimaging.utils.dj_utils import get_primary_key
import datajoint as dj

from djimaging.utils.math_utils import normalize_amp_one
from djimaging.utils.snippet_utils import split_trace_by_reps

BADEN16_DT = 0.128  # 1 / 7.8125 Hz
BADEN16_N_BAR = 32  # Number of time points of the moving bar time component
MIN_N_SHUFFLES = 1000  # Shuffles of the DS permutation test used for the training data


class Baden16TracesV2Template(dj.Computed):
    database = ""

    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'

    @property
    def definition(self):
        definition = """
        # Process traces to match Baden 2016 format, e.g. for RGC classifier.
        -> self.traces_table().proj(chirp_stim_name='stim_name')
        -> self.osds_table().proj(bar_stim_name='stim_name')
        ---
        preproc_chirp: <blob>  # preprocessed chirp trace (averaged, downsampled and normalized)
        preproc_bar:   <blob>  # moving bar time component of OsDsIndexes (first 32 samples, normalized)
        ds_pvalue:     float32 # DS p-value of OsDsIndexes, classifier feature
        """
        return definition

    @property
    def key_source(self):
        try:
            return ((self.traces_table & dict(stim_name=self._stim_name_chirp)).proj(chirp_stim_name='stim_name') *
                    (self.osds_table & dict(stim_name=self._stim_name_bar)).proj(bar_stim_name='stim_name'))
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

    @property
    @abstractmethod
    def osds_table(self):
        """OsDsIndexes table (moving bar time component and DS p-value)."""
        pass

    def make(self, key):
        # Fetch chirp and preprocess
        chirp_stim_dict, chirp_ntrigger_rep = (self.stimulus_table() & dict(stim_name=self._stim_name_chirp)).fetch1(
            'stim_dict', 'ntrigger_rep')
        chirp_triggertimes = (self.presentation_table & dict(stim_name=self._stim_name_chirp) & key).fetch1(
            'triggertimes')
        chirp_trace, chirp_t0, chirp_dt = (self.traces_table & dict(stim_name=self._stim_name_chirp) & key).fetch1(
            'trace', 'trace_t0', 'trace_dt')
        chirp_triggertimes, chirp_trace = load_array(chirp_triggertimes), load_array(chirp_trace)

        qi, chirp_average = preprocess_chirp_v2(chirp_trace, chirp_t0, chirp_dt, chirp_triggertimes, chirp_ntrigger_rep)

        # Fetch moving bar features of OsDsIndexes
        bar_snippet_base_dt = (self.stimulus_table() & dict(stim_name=self._stim_name_bar)).fetch1('snippet_base_dt')
        time_component, time_component_dt, ds_pvalue = (
                self.osds_table & dict(stim_name=self._stim_name_bar) & key).fetch1(
            'time_component', 'time_component_dt', 'ds_pvalue')
        check_bar_features_match_classifier(
            n_shuffles=self.osds_table._n_shuffles, snippet_base_dt=bar_snippet_base_dt, dt=time_component_dt)

        key = key.copy()
        key['preproc_chirp'] = chirp_average
        key['preproc_bar'] = preprocess_bar_v2(load_array(time_component))
        key['ds_pvalue'] = ds_pvalue

        self.insert1(key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        preproc_chirp = load_array((self & key).fetch1('preproc_chirp'))
        preproc_bar = load_array((self & key).fetch1('preproc_bar'))

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

    baden16like_average = interp1d(
        average_times + 0.128, average, assume_sorted=True, bounds_error=False,
        fill_value=(average[0], average[-1]))(baden16like_times)

    # Normalize
    baden16like_average /= np.max(np.abs(baden16like_average))

    return qi, baden16like_average


def check_bar_features_match_classifier(n_shuffles: int, snippet_base_dt, dt: float) -> None:
    """Raise if the OsDsIndexes features were not computed like the training data of the classifier."""
    if n_shuffles < MIN_N_SHUFFLES:
        raise ValueError(f"OsDsIndexes._n_shuffles={n_shuffles}, but the classifier needs >= {MIN_N_SHUFFLES} "
                         f"shuffles for ds_pvalue.")
    if snippet_base_dt is None or not np.isclose(snippet_base_dt, MOVINGBAR_SNIPPET_BASE_DT):
        raise ValueError(f"Stimulus.snippet_base_dt={snippet_base_dt} of the moving bar, but the classifier needs the "
                         f"Baden et al. 2016 per-trial baseline of {MOVINGBAR_SNIPPET_BASE_DT:.3f} s.")
    if not np.isclose(dt, BADEN16_DT, rtol=0.02):
        raise ValueError(f"Moving bar snippets have dt={dt}, but the classifier needs {BADEN16_DT} s (7.8125 Hz).")


def preprocess_bar_v2(time_component: np.ndarray) -> np.ndarray:
    """Return the first 32 samples (at 7.8125 Hz) of the OsDsIndexes time component, normalized to max. abs. 1."""
    baden16like_average = np.array(time_component[:BADEN16_N_BAR], dtype=float)
    baden16like_average /= np.max(np.abs(baden16like_average))
    return baden16like_average
