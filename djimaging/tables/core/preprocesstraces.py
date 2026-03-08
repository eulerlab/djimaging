import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from djimaging.utils import filter_utils, math_utils, plot_utils, trace_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.filter_utils import lowpass_filter_trace
from djimaging.utils.plot_utils import plot_trace_and_trigger, prep_long_title


class PreprocessParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.stimulus_table
        preprocess_id:       tinyint unsigned    # unique param set id
        ---
        window_length:       int       # window length for SavGol filter in seconds
        poly_order:          int       # order of polynomial for savgol filter
        non_negative:        tinyint unsigned  # Clip negative values of trace
        subtract_baseline:   tinyint unsigned  # Subtract baseline
        standardize:         tinyint unsigned  # standardize (1: with sd of baseline, 2: sd of trace, 0: nothing)
        f_cutoff = 0 : float  # Cutoff frequency for low pass filter, only applied when > 0.
        fs_resample = 0 : float  # Resampling frequency, only applied when > 0.
        """
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def add_default(
            self,
            preprocess_id: int = 1,
            stim_names: list | None = None,
            window_length: int = 60,
            poly_order: int = 3,
            non_negative: bool = False,
            subtract_baseline: bool = True,
            standardize: int = 1,
            f_cutoff: float | None = None,
            fs_resample: float | None = None,
            skip_duplicates: bool = False,
    ) -> None:
        """Add default preprocessing parameters for all (or specified) stimuli.

        Parameters
        ----------
        preprocess_id : int, optional
            Unique identifier for this parameter set. Default is 1.
        stim_names : list | None, optional
            List of stimulus names to add parameters for. If None, uses all
            stimuli in the stimulus table. Default is None.
        window_length : int, optional
            Window length in seconds for the Savitzky-Golay filter. Default is 60.
        poly_order : int, optional
            Polynomial order for the Savitzky-Golay filter. Default is 3.
        non_negative : bool, optional
            If True, clip negative values of the trace. Default is False.
        subtract_baseline : bool, optional
            If True, subtract the pre-stimulus baseline. Default is True.
        standardize : int, optional
            Standardization mode: 0 = none, 1 = by baseline std, 2 = by trace std.
            Default is 1.
        f_cutoff : float | None, optional
            Low-pass filter cutoff frequency in Hz. Applied only when > 0.
            Default is None (no filtering).
        fs_resample : float | None, optional
            Target resampling frequency in Hz. Applied only when > 0.
            Default is None (no resampling).
        skip_duplicates : bool, optional
            If True, silently skip duplicate entries. Default is False.
        """
        if stim_names is None:
            stim_names = (self.stimulus_table()).fetch('stim_name')

        key = dict(
            preprocess_id=preprocess_id,
            window_length=window_length,
            poly_order=int(poly_order),
            non_negative=int(non_negative),
            subtract_baseline=int(subtract_baseline),
            standardize=int(standardize),
            f_cutoff=f_cutoff if f_cutoff is not None else 0,
            fs_resample=fs_resample if fs_resample is not None else 0,
        )

        for stim_name in stim_names:
            """Add default preprocess parameter to table"""
            stim_key = key.copy()
            stim_key['stim_name'] = stim_name
            self.insert1(stim_key, skip_duplicates=skip_duplicates)


class PreprocessTracesTemplate(dj.Computed):
    database = ""
    _baseline_max_dt = np.inf

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.traces_table
        -> self.preprocessparams_table
        ---
        pp_trace: longblob    # preprocessed trace
        smoothed_trace:   longblob    # output of savgol filter which is subtracted from the raw trace
        pp_trace_t0:         float       # numerical array of trace times
        pp_trace_dt:         float       # time between frames
        """
        return definition

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    @abstractmethod
    def preprocessparams_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            return ((self.traces_table() & 'trace_valid=1' & 'trigger_valid=1') * self.preprocessparams_table()).proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key: dict) -> None:
        """Preprocess a raw trace and store the result.

        Fetches preprocessing parameters and the raw trace, applies
        detrending, optional baseline subtraction, optional standardisation,
        optional low-pass filtering and optional resampling, then inserts the
        processed trace.

        Parameters
        ----------
        key : dict
            The primary key identifying the entry to populate.
        """
        window_len_seconds, poly_order, subtract_baseline, non_negative, standardize, f_cutoff, fs_resample = \
            (self.preprocessparams_table() & key).fetch1(
                'window_length', 'poly_order', 'subtract_baseline', 'non_negative', 'standardize',
                'f_cutoff', 'fs_resample')

        trace, trace_t0, trace_dt = (self.traces_table() & key).fetch1('trace', 'trace_t0', 'trace_dt')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')

        stim_start = triggertimes[0] if len(triggertimes) > 0 else None

        if stim_start is None and (subtract_baseline or standardize):
            raise ValueError(
                f"No triggers found for {key}, cannot compute baseline. "
                f"Use preprocessing without baseline or fix triggers.")

        pp_trace, smoothed_trace, pp_trace_dt = process_trace(
            trace=trace, trace_t0=trace_t0, trace_dt=trace_dt,
            stim_start=stim_start, poly_order=poly_order, window_len_seconds=window_len_seconds,
            subtract_baseline=subtract_baseline, standardize=standardize, non_negative=non_negative,
            f_cutoff=f_cutoff, fs_resample=fs_resample, baseline_max_dt=self._baseline_max_dt)

        self.insert1(dict(
            key,
            pp_trace_t0=trace_t0,
            pp_trace_dt=pp_trace_dt,
            pp_trace=pp_trace.astype(np.float32),
            smoothed_trace=smoothed_trace.astype(np.float32)
        ))

    def gui_clip_trace(self, key: dict) -> None:
        """Launch an interactive GUI for clipping a preprocessed trace.

        Note that this can not be easily undone for now.

        Parameters
        ----------
        key : dict
            Primary key identifying the trace to clip.
        """

        trace, smoothed_trace, trace_t0, trace_dt = (self & key).fetch1(
            'pp_trace', 'smoothed_trace', 'pp_trace_t0', 'pp_trace_dt')
        trace_t = np.arange(trace.size) * trace_dt + trace_t0

        import ipywidgets as widgets

        w_left = widgets.IntSlider(0, min=0, max=trace.size - 1, step=1,
                                   layout=widgets.Layout(width='800px'))
        w_right = widgets.IntSlider(trace.size - 1, min=0, max=trace.size - 1, step=1,
                                    layout=widgets.Layout(width='800px'))
        w_save = widgets.Checkbox(False)

        title = 'Not saved\n' + prep_long_title(key)

        @widgets.interact(left=w_left, right=w_right, save=w_save)
        def plot_fit(left=0, right=trace.size - 1, save=False):
            nonlocal title

            plot_left_right_clipping(trace, trace_t, left, right, title)

            if save:
                i0, i1 = (right, left + 1) if right < left else (left, right + 1)
                self.update_key_from_gui(
                    key, trace=trace[i0:i1], smoothed_trace=smoothed_trace[i0:i1],
                    trace_t0=trace_t0 + left * trace_dt, trace_dt=trace_dt)
                title = f'SAVED: left={left}, right={right}\n{prep_long_title(key)}'
                w_save.value = False

    def update_key_from_gui(
            self,
            key: dict,
            trace: np.ndarray,
            smoothed_trace: np.ndarray,
            trace_t0: float,
            trace_dt: float,
    ) -> None:
        """Update an existing preprocessed trace entry after GUI editing.

        Parameters
        ----------
        key : dict
            Primary key identifying the entry to update.
        trace : np.ndarray
            Updated preprocessed trace.
        smoothed_trace : np.ndarray
            Updated smoothed (trend) trace.
        trace_t0 : float
            Start time of the updated trace.
        trace_dt : float
            Sampling interval of the updated trace in seconds.
        """
        entry = dict(**key, pp_trace=trace, smoothed_trace=smoothed_trace, pp_trace_t0=trace_t0, pp_trace_dt=trace_dt)
        self.update1(entry)

    def plot1(self, key: dict | None = None, xlim: tuple | None = None, ylim: tuple | None = None) -> None:
        """Plot raw trace, trend, and preprocessed trace for a single entry.

        Parameters
        ----------
        key : dict | None, optional
            Primary key identifying the entry to plot. If None, the first
            available key is used.
        xlim : tuple | None, optional
            x-axis limits. Default is None (auto).
        ylim : tuple | None, optional
            y-axis limits for the preprocessed trace panel. Default is None (auto).
        """
        key = get_primary_key(self, key)

        pp_trace_t0, pp_trace_dt, pp_trace, smoothed_trace = (self & key).fetch1(
            "pp_trace_t0", "pp_trace_dt", "pp_trace", "smoothed_trace")
        trace_t0, trace_dt, trace = (self.traces_table() & key).fetch1("trace_t0", "trace_dt", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        trace_times = np.arange(trace.size) * trace_dt + pp_trace_t0
        pp_trace_times = np.arange(pp_trace.size) * pp_trace_dt + pp_trace_t0

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex='all')

        ax = axs[0]
        plot_trace_and_trigger(time=trace_times, trace=trace - np.mean(smoothed_trace),
                               triggertimes=triggertimes, ax=ax, title=str(key))
        ax.set(ylabel='mean subtracted\nraw trace')

        ax = axs[1]
        plot_trace_and_trigger(time=trace_times, trace=smoothed_trace - np.mean(smoothed_trace),
                               triggertimes=triggertimes, ax=ax)
        ax.set(ylabel='mean subtracted\ntrend')

        ax = axs[2]
        plot_trace_and_trigger(time=pp_trace_times, trace=pp_trace, triggertimes=triggertimes, ax=ax)
        ax.set(ylabel='preprocessed\ntrace')

        ax.set(xlim=xlim, ylim=ylim)

        plt.show()

    def plot(self, restriction: dict | None = None, sort: bool = True) -> None:
        """Plot a heatmap of all preprocessed traces matching the restriction.

        Parameters
        ----------
        restriction : dict | None, optional
            Restriction applied before fetching traces. Default is None (all).
        sort : bool, optional
            Whether to sort traces before plotting. Default is True.
        """
        if restriction is None:
            restriction = dict()

        preprocess_traces = (self & restriction).fetch("pp_trace")
        preprocess_traces = math_utils.padded_vstack(preprocess_traces, cval=np.nan)
        n = preprocess_traces.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 1 + np.minimum(n * 0.1, 10)))
        if len(restriction) > 0:
            plot_utils.set_long_title(fig=fig, title=restriction)

        sort_idxs = trace_utils.argsort_traces(preprocess_traces, ignore_nan=True) if sort else np.arange(n)

        ax.set_title('preprocess_traces')
        plot_utils.plot_signals_heatmap(ax=ax, signals=preprocess_traces[sort_idxs, :], symmetric=False)
        plt.show()


def drop_left_and_right(
        trace: np.ndarray,
        drop_nmin_lr: tuple = (0, 0),
        drop_nmax_lr: tuple = (3, 3),
        inplace: bool = False,
) -> np.ndarray:
    """Drop left and right most values if they are out of limits of the rest of the trace.

    This is necessary because the first few and last few points may be affected
    by artifacts. This is relatively conservative.

    Parameters
    ----------
    trace : np.ndarray
        1-D trace array to process.
    drop_nmin_lr : tuple, optional
        Minimum number of samples to replace at the (left, right) ends.
        Default is (0, 0).
    drop_nmax_lr : tuple, optional
        Maximum number of samples to examine for out-of-range replacement at
        the (left, right) ends. Default is (3, 3).
    inplace : bool, optional
        If True, modify `trace` in place. Default is False.

    Returns
    -------
    np.ndarray
        Trace with edge artifacts replaced by the nearest non-artifact value.
    """
    if not inplace:
        trace = trace.copy()

    if (drop_nmax_lr[0] > 0) or (drop_nmax_lr[1] > 0):
        l_idx = drop_nmax_lr[0]
        r_idx = -drop_nmax_lr[1] if drop_nmax_lr[1] > 0 else trace.size
        trace_min = np.min(trace[l_idx:r_idx])
        trace_max = np.max(trace[l_idx:r_idx])
    else:
        l_idx, r_idx, trace_min, trace_max = None, None, None, None

    if drop_nmax_lr[0] > 0:
        l_oor = (trace[:l_idx] < trace_min) | (trace[:l_idx] > trace_max)
        if np.all(l_oor):
            l_ndrop = drop_nmax_lr[0]
        else:
            l_ndrop = np.argmin(l_oor)
        if l_ndrop > 0:
            trace[:l_ndrop] = trace[l_ndrop]

    if drop_nmax_lr[1] > 0:
        r_oor = ((trace[r_idx:] < trace_min) | (trace[r_idx:] > trace_max))
        if np.all(r_oor):
            r_ndrop = drop_nmax_lr[1]
        else:
            r_ndrop = np.argmin(r_oor[::-1])
        if r_ndrop > 0:
            trace[-r_ndrop:] = trace[-(r_ndrop + 1)]

    if drop_nmin_lr[0] > 0:
        trace[:drop_nmin_lr[0]] = trace[drop_nmin_lr[0]]
    if drop_nmin_lr[1] > 0:
        trace[-drop_nmin_lr[1]:] = trace[-drop_nmin_lr[1]]

    return trace


def detrend_trace(
        trace: np.ndarray,
        window_len_seconds: float,
        fs: float,
        poly_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend a trace using a Savitzky-Golay filter.

    Parameters
    ----------
    trace : np.ndarray
        1-D trace array.
    window_len_seconds : float
        Length of the filter window in seconds.
    fs : float
        Sampling frequency in Hz.
    poly_order : int
        Polynomial order for the Savitzky-Golay filter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        detrended_trace : np.ndarray
            Trace with the smooth trend subtracted.
        smoothed_trace : np.ndarray
            Estimated smooth trend (output of the Savitzky-Golay filter).
    """
    window_len_frames = np.ceil(window_len_seconds * fs)

    if window_len_frames % 2 == 0:
        window_len_frames -= 1

    window_len_frames = int(window_len_frames)
    smoothed_trace = signal.savgol_filter(trace, window_length=window_len_frames, polyorder=poly_order, mode='nearest')
    detrended_trace = trace - smoothed_trace
    return detrended_trace, smoothed_trace


def extract_baseline(
        trace: np.ndarray,
        trace_times: np.ndarray,
        stim_start: float,
        max_dt: float = np.inf,
) -> np.ndarray:
    """Compute baseline for trace by extracting pre-stimulus samples.

    Parameters
    ----------
    trace : np.ndarray
        1-D trace array.
    trace_times : np.ndarray
        Time stamps corresponding to each sample in `trace`.
    stim_start : float
        Stimulus onset time. Baseline is taken from samples before this time.
    max_dt : float, optional
        Maximum duration of the baseline window in seconds (measured backwards
        from `stim_start`). Default is np.inf (use all pre-stimulus samples).

    Returns
    -------
    np.ndarray
        Baseline samples extracted from `trace`.

    Raises
    ------
    ValueError
        If no samples in `trace_times` occur before `stim_start`.
    """
    if not np.any(trace_times < stim_start):
        raise ValueError(f"stim_start={stim_start:.1g}, trace_start={trace_times.min():.1g}")

    baseline_end = np.nonzero(trace_times[trace_times < stim_start])[0][-1]
    baseline_start = np.nonzero(trace_times >= trace_times[baseline_end] - max_dt)[0][0]
    baseline = trace[baseline_start:baseline_end]
    return baseline


def non_negative_trace(trace: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Make trace non-negative by removing the lowest 2.5 percentile.

    Parameters
    ----------
    trace : np.ndarray
        1-D trace array.
    inplace : bool, optional
        If True, modify `trace` in place. Default is False.

    Returns
    -------
    np.ndarray
        Trace shifted so that its minimum is zero.
    """
    if not inplace:
        trace = trace.copy()

    clip_value = np.percentile(trace, q=2.5)
    trace[trace < clip_value] = clip_value
    trace -= clip_value

    return trace


def subtract_baseline_trace(
        trace: np.ndarray,
        trace_times: np.ndarray,
        stim_start: float,
        inplace: bool = False,
        max_dt: float = np.inf,
) -> np.ndarray:
    """Subtract the median pre-stimulus baseline from a trace.

    Parameters
    ----------
    trace : np.ndarray
        1-D trace array.
    trace_times : np.ndarray
        Time stamps corresponding to each sample in `trace`.
    stim_start : float
        Stimulus onset time; baseline is taken from samples before this.
    inplace : bool, optional
        If True, modify `trace` in place. Default is False.
    max_dt : float, optional
        Maximum baseline window duration in seconds. Default is np.inf.

    Returns
    -------
    np.ndarray
        Trace with its pre-stimulus median subtracted.
    """
    if not inplace:
        trace = trace.copy()

    baseline = extract_baseline(trace, trace_times, stim_start, max_dt=max_dt)
    if len(baseline) == 0:
        warnings.warn(f"No baseline found for stim_start={stim_start:.1f}, trace_t0={trace_times[0]:.1f}")
    else:
        trace -= np.median(baseline)

    return trace


def process_trace(
        trace: np.ndarray,
        trace_t0: float,
        trace_dt: float,
        poly_order: int = 3,
        window_len_seconds: float = 0,
        subtract_baseline: bool = False,
        standardize: int = 0,
        non_negative: bool = False,
        stim_start: float | None = None,
        f_cutoff: float | None = None,
        fs_resample: float | None = None,
        drop_nmin_lr: tuple = (0, 0),
        drop_nmax_lr: tuple = (3, 3),
        baseline_max_dt: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detrend and preprocess a trace.

    Applies, in order: edge-artifact removal, optional low-pass filtering,
    optional Savitzky-Golay detrending, optional baseline subtraction,
    optional non-negativity clipping, optional standardization, and optional
    resampling.

    Parameters
    ----------
    trace : np.ndarray
        1-D raw trace array.
    trace_t0 : float
        Time of the first sample in seconds.
    trace_dt : float
        Sampling interval in seconds.
    poly_order : int, optional
        Polynomial order for the Savitzky-Golay filter. Default is 3.
    window_len_seconds : float, optional
        Window length in seconds for the Savitzky-Golay filter. If 0, no
        detrending is applied. Default is 0.
    subtract_baseline : bool, optional
        If True, subtract the pre-stimulus median. Default is False.
    standardize : int, optional
        0 = none, 1 = divide by baseline std, 2 = divide by trace std.
        Default is 0.
    non_negative : bool, optional
        If True, clip and shift trace to be non-negative. Default is False.
    stim_start : float | None, optional
        Stimulus onset time used for baseline estimation. Default is None.
    f_cutoff : float | None, optional
        Low-pass filter cutoff frequency in Hz. Applied only when > 0.
        Default is None.
    fs_resample : float | None, optional
        Target resampling frequency in Hz. Applied only when > 0. Default is None.
    drop_nmin_lr : tuple, optional
        Minimum samples to replace at (left, right) edges. Default is (0, 0).
    drop_nmax_lr : tuple, optional
        Maximum samples to examine for edge replacement. Default is (3, 3).
    baseline_max_dt : float, optional
        Maximum baseline window duration in seconds. Default is np.inf.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        trace : np.ndarray
            Preprocessed trace.
        smoothed_trace : np.ndarray
            Smooth trend that was subtracted (zeros if no detrending).
        trace_dt_new : float
            Sampling interval of the returned trace (may differ if resampled).

    Raises
    ------
    ValueError
        If `standardize` is not in [0, 1, 2], or if `stim_start` is None
        when required by `subtract_baseline` or `standardize`.
    """
    assert standardize in [0, 1, 2], standardize

    trace_times = np.arange(trace.size) * trace_dt + trace_t0
    fs = 1. / trace_dt

    trace = np.asarray(trace).copy()

    trace = drop_left_and_right(trace, drop_nmin_lr=drop_nmin_lr, drop_nmax_lr=drop_nmax_lr, inplace=True)

    if (f_cutoff is not None) and (f_cutoff > 0):
        trace = lowpass_filter_trace(trace=trace, fs=fs, f_cutoff=f_cutoff)

    if window_len_seconds > 0:
        trace, smoothed_trace = detrend_trace(trace, window_len_seconds, fs, poly_order)
    else:
        smoothed_trace = np.zeros_like(trace)

    if stim_start is not None:
        if not np.any(trace_times < stim_start):
            raise ValueError(f"stim_start={stim_start:.1g}, trace_start={trace_times.min():.1g}")

    if subtract_baseline:
        if stim_start is None:
            raise ValueError("stim_start must be provided to subtract baseline. probably there are no triggers")
        trace = subtract_baseline_trace(trace, trace_times, stim_start, max_dt=baseline_max_dt, inplace=True)

    if non_negative:
        trace = non_negative_trace(trace, inplace=True)

    if standardize == 1:
        if stim_start is None:
            raise ValueError("stim_start must be provided to standardize. probably there are no triggers")
        baseline = extract_baseline(trace, trace_times, stim_start)
        trace /= np.std(baseline)
    elif standardize == 2:
        trace /= np.std(trace)

    if (fs_resample is not None) and (fs_resample != 0) and (fs != fs_resample):
        if (fs_resample < fs) and np.isclose((fs / fs_resample), np.round(fs / fs_resample)):
            fdownsample = int(np.round(fs / fs_resample))
            trace_dt_new = trace_dt * fdownsample
            trace_times, trace = filter_utils.downsample_trace(
                tracetime=trace_times, trace=trace, fdownsample=fdownsample)
        else:
            trace_dt_new = 1. / fs_resample
            trace_times, trace = filter_utils.resample_trace(tracetime=trace_times, trace=trace, dt=trace_dt_new)
    else:
        trace_dt_new = trace_dt

    return trace, smoothed_trace, trace_dt_new


def plot_left_right_clipping(
        trace_: np.ndarray,
        trace_t_: np.ndarray,
        left_: int,
        right_: int,
        title_: str,
) -> None:
    """Plot a trace with vertical lines indicating the selected clip boundaries.

    Parameters
    ----------
    trace_ : np.ndarray
        1-D trace array to display.
    trace_t_ : np.ndarray
        Time stamps corresponding to each sample in `trace_`.
    left_ : int
        Index of the left clip boundary.
    right_ : int
        Index of the right clip boundary.
    title_ : str
        Title string for the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_title(title_)
    ax.plot(trace_t_, trace_)
    ax.axvline(trace_t_[left_], c='r', ls='--')
    ax.axvline(trace_t_[right_], c='r', ls='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trace')
    plt.show()
