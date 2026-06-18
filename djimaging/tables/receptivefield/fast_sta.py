"""Fast STA table template for receptive field estimation

Example usage:
@schema
class FastStaParams(receptivefield.FastStaParamsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation

@schema
class FastSta(receptivefield.FastStaTemplate):
    params_table = FastStaParams
    presentation_table = Presentation
    traces_table = PreprocessTraces
"""

from abc import abstractmethod

import datajoint as dj
from matplotlib import pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.receptive_fields.fit_rf_utils import build_design_matrix, get_rf_timing_params, compute_rf_sta
from djimaging.utils.receptive_fields.plot_rf_utils import plot_rf_frames, plot_rf_video
from djimaging.utils.receptive_fields.preprocess_rf_utils import preprocess_stimulus, finalize_stim, \
    preprocess_trace, finalize_trace
from djimaging.utils.trace_utils import get_mean_dt, align_trace_to_stim


class FastStaParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.stimulus_table
        dnoise_params_id: tinyint unsigned # unique param set id
        ---
        x_stimulus : longblob  # Stimulus design matrix
        fit_kind : varchar(191)
        fupsample_trace : tinyint unsigned  # Multiplier of sampling frequency, using linear interpolation.
        fupsample_stim = 0 : tinyint unsigned  # Multiplier of sampling stimulus, using repeat.
        lowpass_cutoff = 0: float  # Cutoff frequency low pass filter, applied if larger 0.
        pre_blur_sigma_s = 0: float  # Gaussian blur applied after low pass filter.
        post_blur_sigma_s = 0: float  # Gaussian blur applied after all other steps.
        filter_dur_s_past : float # filter duration in seconds into the past
        filter_dur_s_future : float # filter duration in seconds into the future
        rf_time: longblob #  time of RF, depends on dt and shift
        dt: float  # Time step between frames
        shift: int  # Shift of stimulus relative to trace. If negative, prediction looks into future.
        dims: blob  # Dimensions of stimulus
        burn_in: int  # Number of frames to burn in
        """
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    def add_default(
            self, stim_names: list | None = None, dnoise_params_id: int = 1, fit_kind: str = "gradient",
            fupsample_trace: int = 10, fupsample_stim: int = 10, lowpass_cutoff: float = 0,
            pre_blur_sigma_s: float = 0, post_blur_sigma_s: float = 0,
            filter_dur_s_past: float = 1., filter_dur_s_future: float = 0.,
            skip_duplicates: bool = False, verbose: bool = True) -> None:
        """Add default preprocess parameter to table.

        Parameters
        ----------
        stim_names : list or None, optional
            Names of noise stimuli to add parameters for. If None, all noise stimuli are used.
        dnoise_params_id : int, optional
            Unique parameter set ID. Default is 1.
        fit_kind : str, optional
            Kind of fitting procedure. Default is "gradient".
        fupsample_trace : int, optional
            Multiplier of sampling frequency using linear interpolation. Default is 10.
        fupsample_stim : int, optional
            Multiplier of sampling stimulus using repeat. Default is 10.
        lowpass_cutoff : float, optional
            Cutoff frequency for low pass filter; applied if larger than 0. Default is 0.
        pre_blur_sigma_s : float, optional
            Gaussian blur sigma applied after low pass filter (in seconds). Default is 0.
        post_blur_sigma_s : float, optional
            Gaussian blur sigma applied after all other steps (in seconds). Default is 0.
        filter_dur_s_past : float, optional
            Filter duration in seconds into the past. Default is 1.0.
        filter_dur_s_future : float, optional
            Filter duration in seconds into the future. Default is 0.0.
        skip_duplicates : bool, optional
            If True, skip duplicate entries. Default is False.
        verbose : bool, optional
            If True, print progress information. Default is True.
        """

        if stim_names is None:
            stim_names = (self.stimulus_table() & "stim_family='noise'").fetch('stim_name')

        key = dict(dnoise_params_id=dnoise_params_id, fit_kind=fit_kind,
                   fupsample_trace=fupsample_trace, fupsample_stim=fupsample_stim,
                   lowpass_cutoff=lowpass_cutoff,
                   pre_blur_sigma_s=pre_blur_sigma_s, post_blur_sigma_s=post_blur_sigma_s,
                   filter_dur_s_past=filter_dur_s_past, filter_dur_s_future=filter_dur_s_future)

        for stim_name in stim_names:
            """Add default preprocess parameter to table"""
            stim_key = key.copy()
            stim_key['stim_name'] = stim_name

            if len(self & stim_key) > 0 and skip_duplicates:
                continue

            stim, stim_dict, ntrigger_rep, isrepeated, framerate = (self.stimulus_table() & stim_key).fetch1(
                "stim_trace", "stim_dict", "ntrigger_rep", "isrepeated", "framerate")

            if framerate > 0:  # This is actually the triggerrate!
                dt_trigger = 1 / framerate
            else:
                triggertimes = (self.presentation_table() & stim_key).fetch('triggertimes')
                dt_trigger = np.median([np.median(np.diff(tti)) for tti in triggertimes])

            if "ntrigger_tot" in stim_dict:
                n_trigger = stim_dict["ntrigger_tot"]
            elif (not isrepeated) and (ntrigger_rep > 1):
                n_trigger = ntrigger_rep
            else:  # Infer from triggertimes
                triggertimes = (self.presentation_table() & stim_key).fetch('triggertimes')
                n_trigger = int(np.percentile([tti.size for tti in triggertimes], q=98))

            stimtime = np.arange(n_trigger) * dt_trigger

            if verbose:
                print(f"Preparing design matrix for stimulus: {stim_name}: "
                      f"{n_trigger=}, {dt_trigger=}, {len(stimtime)=}")

            x_stimulus, rf_time, dims, burn_in, shift, dt, t0 = self.prepare_stimulus(
                stim=stim, triggertimes=stimtime,
                nframes_per_trigger=stim_dict.get('nframes_per_trigger', stim_dict.get('ntrigger_per_frame', 1)),
                # Old name was ntrigger_per_frame
                fupsample_stim=fupsample_stim,
                filter_dur_s_past=filter_dur_s_past,
                filter_dur_s_future=filter_dur_s_future,
            )

            if verbose:
                print(f"Stimulus design matrix has shape: {x_stimulus.shape=} with {dt=}")

            new_key = stim_key.copy()
            new_key['x_stimulus'] = x_stimulus
            new_key['rf_time'] = rf_time
            new_key['dt'] = dt
            new_key['shift'] = shift
            new_key['dims'] = dims
            new_key['burn_in'] = burn_in

            self.insert1(new_key, skip_duplicates=skip_duplicates)

    @staticmethod
    def prepare_stimulus(
            stim: np.ndarray,
            triggertimes: np.ndarray,
            nframes_per_trigger: int,
            fupsample_stim: int,
            filter_dur_s_past: float,
            filter_dur_s_future: float) -> tuple:
        """Preprocess stimulus and build the design matrix.

        Parameters
        ----------
        stim : np.ndarray
            Raw stimulus array.
        triggertimes : np.ndarray
            Array of trigger timestamps.
        nframes_per_trigger : int
            Number of frames per trigger
        fupsample_stim : int
            Upsampling factor for the stimulus.
        filter_dur_s_past : float
            Filter duration in seconds into the past.
        filter_dur_s_future : float
            Filter duration in seconds into the future.

        Returns
        -------
        tuple
            Tuple of (x_stimulus, rf_time, dims, burn_in, shift, dt, t0).
        """
        stimtime, stim = preprocess_stimulus(
            stim, triggertimes,
            nframes_per_trigger=nframes_per_trigger,
            fupsample_stim=fupsample_stim)
        dt, dt_rel_error = get_mean_dt(stimtime, rtol_error=np.inf, rtol_warning=0.5)
        t0 = stimtime[0]

        rf_time, dim_t, shift, burn_in = get_rf_timing_params(filter_dur_s_past, filter_dur_s_future, dt)
        dims = (dim_t,) + stim.shape[1:]

        stim = finalize_stim(stim)
        x_stimulus = build_design_matrix(stim, n_lag=dim_t, shift=shift)

        return x_stimulus, rf_time, dims, burn_in, shift, dt, t0


class FastStaTemplate(dj.Computed):
    database = ""
    _traces_prefix = 'pp_'

    @property
    def definition(self):
        definition = '''
        # Preprocess traces for receptive field estimation
        -> self.params_table
        -> self.traces_table
        ---
        rf: longblob  # spatio-temporal receptive field
        '''
        return definition

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.params_table().proj() * (self.presentation_table().proj() & self.traces_table().proj())
        except (AttributeError, TypeError):
            pass

    def prepare_sta_params(self, *restrictions) -> dict:
        """Fetch and prepare STA parameters from the params table.

        Parameters
        ----------
        *restrictions : dict
            Optional DataJoint restrictions to select a unique parameter set.

        Returns
        -------
        dict
            Dictionary containing all STA parameters needed for computation.

        Raises
        ------
        NotImplementedError
            If the restrictions do not select exactly one parameter set.
        """
        restrictions = [{}] if len(restrictions) == 0 else restrictions
        if len(self.params_table() & [*restrictions]) != 1:
            raise NotImplementedError("Only one parameter set supported. Provide `restrictions` that selects one.")

        (x_stimulus, dt, rf_time, burn_in, shift, kind, dims, fupsample_stim, fupsample_trace,
         fit_kind, lowpass_cutoff, pre_blur_sigma_s, post_blur_sigma_s) = (
                self.params_table & [*restrictions]).fetch1(
            'x_stimulus', 'dt', 'rf_time', 'burn_in', 'shift', 'fit_kind', 'dims', 'fupsample_stim',
            "fupsample_trace", "fit_kind", "lowpass_cutoff", "pre_blur_sigma_s", "post_blur_sigma_s")

        x_stimulus = np.ascontiguousarray(x_stimulus.astype(np.float32))

        # Fetch nframes_per_trigger from the stimulus table so that stimtime
        # reconstructed in make_compute matches the design matrix built in
        # prepare_stimulus. Falls back to the old name 'ntrigger_per_frame'
        # and to 1 if neither is present.
        stim_dict = (self.params_table.stimulus_table & [*restrictions]).fetch1('stim_dict')
        nframes_per_trigger = int(stim_dict.get(
            'nframes_per_trigger', stim_dict.get('ntrigger_per_frame', 1)))

        sta_params = dict(
            x_stimulus=x_stimulus, rf_time=rf_time, burn_in=burn_in, shift=shift, kind=kind, dims=dims,
            dt=dt, fupsample_trace=fupsample_trace, fupsample_stim=fupsample_stim,
            nframes_per_trigger=nframes_per_trigger,
            fit_kind=fit_kind, lowpass_cutoff=lowpass_cutoff,
            pre_blur_sigma_s=pre_blur_sigma_s, post_blur_sigma_s=post_blur_sigma_s,
        )

        return sta_params

    def populate(
            self,
            *restrictions,
            suppress_errors: bool = False,
            return_exception_objects: bool = False,
            reserve_jobs: bool = False,
            order: str = "original",
            limit: int | None = None,
            max_calls: int | None = None,
            display_progress: bool = False,
            processes: int = 1,
            make_kwargs: dict | None = None,
    ) -> None:
        """Populate the table, precomputing STA parameters shared across all entries.

        Parameters
        ----------
        *restrictions : dict
            Optional DataJoint restrictions. Must select exactly one parameter set.
        suppress_errors : bool, optional
            If True, suppress errors during population. Default is False.
        return_exception_objects : bool, optional
            If True, return exception objects instead of raising them. Default is False.
        reserve_jobs : bool, optional
            If True, use DataJoint job reservation. Default is False.
        order : str, optional
            Order in which to populate entries. Default is "original".
        limit : int or None, optional
            Maximum number of entries to populate. Default is None.
        max_calls : int or None, optional
            Maximum number of make() calls. Default is None.
        display_progress : bool, optional
            If True, show a progress bar. Default is False.
        processes : int, optional
            Number of parallel processes. Default is 1.
        make_kwargs : dict or None, optional
            Additional keyword arguments passed to make(). Default is None.

        Raises
        ------
        NotImplementedError
            If the restrictions do not select exactly one parameter set.
        """
        restrictions = [{}] if len(restrictions) == 0 else restrictions
        if len(self.params_table() & [*restrictions]) != 1:
            raise NotImplementedError("Only one parameter set supported. Provide `restrictions` that selects one.")

        if make_kwargs is None:
            make_kwargs = dict()
        make_kwargs['sta_params'] = self.prepare_sta_params(*restrictions)

        super().populate(
            *restrictions,
            suppress_errors=suppress_errors,
            return_exception_objects=return_exception_objects,
            reserve_jobs=reserve_jobs,
            order=order,
            limit=limit,
            max_calls=max_calls,
            display_progress=display_progress,
            processes=processes,
            make_kwargs=make_kwargs,
        )

    def make_compute(self, key: dict, sta_params: dict | None = None) -> list | None:
        """Compute receptive fields for all traces matching the given key.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to compute.
        sta_params : dict or None, optional
            Pre-computed STA parameters. Must be a dict. Default is None.

        Returns
        -------
        list or None
            List of dicts with computed RF entries, or None if no traces found.

        Raises
        ------
        NotImplementedError
            If sta_params is not a dict.
        """
        if isinstance(sta_params, dict):
            x_stimulus = sta_params['x_stimulus']
            dims = sta_params['dims']
            fupsample_trace = sta_params['fupsample_trace']
            fupsample_stim = sta_params['fupsample_stim']
            nframes_per_trigger = sta_params['nframes_per_trigger']
            fit_kind = sta_params['fit_kind']
            lowpass_cutoff = sta_params['lowpass_cutoff']
            pre_blur_sigma_s = sta_params['pre_blur_sigma_s']
            post_blur_sigma_s = sta_params['post_blur_sigma_s']
        else:
            raise NotImplementedError("Only dict supported")

        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')

        # Build stimtime at the same resolution as x_stimulus: each trigger
        # spans nframes_per_trigger stimulus frames, each further upsampled
        # by fupsample_stim. Total upsampling factor per trigger:
        total_upsample = nframes_per_trigger * fupsample_stim

        dt_triggertimes = np.median(np.diff(triggertimes))
        xidxs = np.arange(triggertimes.size + 1)
        xidxs_stim = np.repeat(xidxs, total_upsample) + np.tile(
            np.arange(total_upsample) * 1 / total_upsample, xidxs.size)

        stimtime = np.interp(
            xidxs_stim, xidxs, np.append(triggertimes, triggertimes[-1] + dt_triggertimes))[:-total_upsample]

        trace_t0s, trace_dts, traces, rf_keys = (self.traces_table() & key).fetch(
            self._traces_prefix + 'trace_t0', self._traces_prefix + 'trace_dt', self._traces_prefix + 'trace', 'KEY')

        if len(rf_keys) == 0:
            print(f"No traces found for {key}")
            return

        tracetime_base = np.arange(traces[0].size) * trace_dts[0]

        rf_entries = []
        for trace, trace_t0, rf_key in tqdm(
                zip(traces, trace_t0s, rf_keys), total=len(rf_keys), desc=str(key), leave=False):
            tracetime = tracetime_base + trace_t0
            rf = self._compute_rf(
                tracetime, trace, stimtime, x_stimulus,
                dims, fupsample_trace, fit_kind, lowpass_cutoff, pre_blur_sigma_s, post_blur_sigma_s)

            if np.any(np.isnan(rf)):
                print(f"Skipping {key}, RF contains NaNs. "
                      f"x_stimulus contained {np.sum(np.isnan(x_stimulus))} NaNs. "
                      f"trace contained {np.sum(np.isnan(trace))} NaNs. "
                      f"tracetime contained {np.sum(np.isnan(tracetime))} NaNs. "
                      f"[{tracetime[0]=} ... {tracetime[-1]}] and [{stimtime[0]=} ... {stimtime[-1]}]")
            else:
                rf_entries.append({**key, **rf_key, "rf": rf})

        if len(rf_entries) == 0 and len(traces) > 0:
            print(f"All RFs contain NaNs for {key}, skipping.")

        return rf_entries

    def make(self, key: dict, sta_params: dict | None = None) -> None:
        """Compute and insert receptive fields for all traces matching the given key.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to compute.
        sta_params : dict or None, optional
            Pre-computed STA parameters dictionary. Default is None.
        """
        rf_entries = self.make_compute(key=key, sta_params=sta_params)
        self.insert(rf_entries)

    @staticmethod
    def _compute_rf(
            tracetime: np.ndarray, trace: np.ndarray, stimtime: np.ndarray,
            x_stimulus: np.ndarray, dims: tuple, fupsample_trace: int,
            fit_kind: str, lowpass_cutoff: float, pre_blur_sigma_s: float,
            post_blur_sigma_s: float) -> np.ndarray:
        """Compute the receptive field for a single trace.

        Parameters
        ----------
        tracetime : np.ndarray
            Timestamps corresponding to the trace samples.
        trace : np.ndarray
            1-D neural response trace.
        stimtime : np.ndarray
            Timestamps corresponding to the stimulus frames.
        x_stimulus : np.ndarray
            Stimulus design matrix of shape (n_frames, n_features).
        dims : tuple
            Target shape of the computed receptive field.
        fupsample_trace : int
            Upsampling factor for the trace.
        fit_kind : str
            Kind of fitting procedure (e.g. "gradient").
        lowpass_cutoff : float
            Cutoff frequency for the low-pass filter.
        pre_blur_sigma_s : float
            Gaussian blur sigma applied before STA (in seconds).
        post_blur_sigma_s : float
            Gaussian blur sigma applied after STA (in seconds).

        Returns
        -------
        np.ndarray
            Receptive field array reshaped to `dims`.
        """

        tracetime, trace = preprocess_trace(
            tracetime, trace, fupsample_trace, fit_kind, lowpass_cutoff, pre_blur_sigma_s)
        trace, dt, t0, dt_rel_error = align_trace_to_stim(
            stimtime=stimtime, trace=trace, tracetime=tracetime)
        trace = finalize_trace(trace, dt, post_blur_sigma_s).astype(np.float32)

        if trace.size < x_stimulus.shape[0] * 1 / 3:
            print(f"WARNING: Trace doesn't cover much of the stimulus duration."
                  f"trace-size: {trace.size}, stim-size: {x_stimulus.shape[0]}")

        assert trace.size <= x_stimulus.shape[0], f"stim-shape: {x_stimulus.shape}, trace-shape: {trace.shape}"
        assert trace.dtype == x_stimulus.dtype, (trace.dtype, x_stimulus.dtype)
        rf = compute_rf_sta(np.ascontiguousarray(x_stimulus[:trace.size]),
                            np.ascontiguousarray(trace)).reshape(dims)

        return rf

    def plot1(self, key: dict | None = None, kind: str = 'frames', **kwargs) -> None:
        """Plot the receptive field as frames.

        Parameters
        ----------
        key : dict or None, optional
            DataJoint key to restrict the table. Default is None.
        kind : str, optional
            Kind of plot to show. Currently only 'frames' is supported. Default is 'frames'.
        **kwargs
            Additional keyword arguments passed to the specific plot function.
        """
        if kind == 'frames':
            self.plot1_frames(key=key, **kwargs)
        elif kind == 'video':
            self.plot1_video(key=key, **kwargs)
        elif kind == 'traces':
            self.plot1_traces(key=key, **kwargs)
        else:
            raise NotImplementedError(f"kind {kind} is not supported.")

    def plot1_frames(self, key: dict | None = None, downsample: int = 1) -> None:
        """Plot the receptive field as individual frames.

        Parameters
        ----------
        key : dict or None, optional
            DataJoint key to restrict the table. Default is None.
        downsample : int, optional
            Downsampling factor for the frames. Default is 1.
        """
        key = get_primary_key(table=self, key=key)
        rf_time = (self.params_table & key).fetch1('rf_time')
        rf = (self & key).fetch1('rf')
        plot_rf_frames(rf, rf_time, downsample=downsample)

    def plot1_video(self, key: dict | None = None, fps: int = 10):
        """Plot the receptive field as a video animation.

        Parameters
        ----------
        key : dict or None, optional
            DataJoint key to restrict the table. Default is None.
        fps : int, optional
            Frames per second for the animation. Default is 10.

        Returns
        -------
        matplotlib.animation.Animation
            Animation object of the RF video.
        """
        key = get_primary_key(table=self, key=key)
        rf_time = (self.params_table & key).fetch1('rf_time')
        rf = (self & key).fetch1('rf')
        return plot_rf_video(rf, rf_time, fps=fps)

    def plot1_traces(self, key: dict | None = None, ax=None, cmap: str = 'viridis'):
        """Plot the spatial components of the RF as lines vs. lag.

        Useful for few-pixel stimuli where the spatial structure is low-dimensional
        (e.g. 1D bar noise or small 2D grids). Each spatial pixel is plotted as a
        line over rf_time, color-coded by its position along the flattened spatial
        axis.

        Parameters
        ----------
        key : dict or None, optional
            DataJoint key to restrict the table. Default is None.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot into. If None, a new figure and axes are created.
        cmap : str, optional
            Matplotlib colormap name used to color pixels by spatial position.
            Default is 'viridis'.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        key = get_primary_key(table=self, key=key)
        rf_time = (self.params_table & key).fetch1('rf_time')
        rf = (self & key).fetch1('rf')

        # Reshape to (n_lags, n_pixels), handling both 2D and 3D+ RFs.
        n_lags = rf.shape[0]
        rf_flat = rf.reshape(n_lags, -1)
        n_pixels = rf_flat.shape[1]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_pixels))
        for i in range(n_pixels):
            ax.plot(rf_time, rf_flat[:, i], color=colors[i], lw=1)

        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('RF amplitude')
        ax.set_title(f"RF traces ({n_pixels} pixels)")

        return ax


class FastStaQualityTemplate(dj.Computed):
    database = ""
    # Duration (in seconds) at the acausal end of rf_time used as the noise/baseline region.
    # For shift <= 0 (filter into the past), this is the most-future portion of rf_time,
    # which should contain no stimulus-driven signal.
    _baseline_dur_s = 0.1

    @property
    def definition(self):
        definition = """
        # Signal-to-noise ratio for STA receptive fields (peak-to-baseline at best pixel).
        -> self.sta_table
        ---
        snr: float              # peak_amplitude / noise_std at the best spatial location
        peak_amplitude = NULL: float   # max(|rf|) at the best spatial location, over all time
        peak_z = NULL : float
        noise_std : float        # std of rf at the best spatial location, within the baseline window
        n_baseline_frames: int unsigned  # number of time bins used as baseline
        """
        return definition

    @property
    @abstractmethod
    def sta_table(self):
        pass

    def make(self, key: dict) -> None:
        rf = (self.sta_table & key).fetch1('rf')
        rf_time = (self.sta_table.params_table & key).fetch1('rf_time')
     
        # Flatten spatial/feature dims so rf becomes (T, S) regardless of input shape.
        T = rf.shape[0]
        rf_flat = rf.reshape(T, -1)  # (T, S); S == 1 if rf was 1D
        S = rf_flat.shape[1]
     
        # Determine baseline region: acausal end of rf_time.
        # rf_time is ordered from past to future, so the most-future bins are at the end.
        dt = np.median(np.diff(rf_time))
        n_baseline = max(1, int(round(self._baseline_dur_s / dt)))
        n_baseline = min(n_baseline, T - 1)  # leave at least one signal frame
     
        # Pooled noise estimate across ALL spatial locations in the acausal tail.
        # This decouples noise estimation from pixel selection and makes the
        # estimate stable and (asymptotically) invariant to adding noise-only
        # pixels via spatial padding outside the RF.
        baseline = rf_flat[-n_baseline:, :]  # (n_baseline, S)
        noise_std = float(np.std(baseline))
     
        # Raw peak across all space and time.
        peak = float(np.max(np.abs(rf_flat)))
     
        # Extreme-value correction. With S pixels and T lags of approximately
        # independent Gaussian noise, E[max |z|] ~ sqrt(2 * log(T * S)). Dividing
        # the raw peak-z by this expected null peak yields a statistic whose null
        # distribution is roughly centered at 1 regardless of S (and T), so
        # padding the stimulus with noise-only pixels no longer inflates SNR.
        if noise_std > 0:
            peak_z = peak / noise_std
            null_peak = np.sqrt(2.0 * np.log(max(T * S, 2)))
            snr = peak_z / null_peak
        else:
            peak_z = np.nan
            snr = np.nan
     
        # Argmax pixel kept only for reporting / downstream use.
        abs_rf = np.abs(rf_flat)
        best_pixel = int(np.argmax(abs_rf.max(axis=0)))
     
        self.insert1({
            **key,
            'snr': snr,                       # size-invariant: peak_z / E[max|z|] under null
            'peak_amplitude': peak,           # raw peak (same units as rf)
            'peak_z': peak_z,                 # peak in units of pooled noise std
            'noise_std': noise_std,           # pooled across all S pixels
            'n_baseline_frames': n_baseline,
        })


    def plot1(self, key: dict | None = None, ax=None):
        """Plot the RF traces with the baseline window highlighted.

        Shows all spatial components as lines over rf_time, shades the baseline
        region used for noise estimation, and marks the peak amplitude. The
        computed SNR is shown in the title.

        Parameters
        ----------
        key : dict or None, optional
            DataJoint key to restrict the table. Default is None.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot into. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        from matplotlib import pyplot as plt

        key = get_primary_key(table=self, key=key)

        snr, peak_amplitude, noise_std, n_baseline = (self & key).fetch1(
            'snr', 'peak_amplitude', 'noise_std', 'n_baseline_frames')
        rf = (self.sta_table & key).fetch1('rf')
        rf_time = (self.sta_table.params_table & key).fetch1('rf_time')

        # Flatten spatial dims to (n_lags, n_pixels)
        n_lags = rf.shape[0]
        rf_flat = rf.reshape(n_lags, -1)
        n_pixels = rf_flat.shape[1]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))

        colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_pixels))
        for i in range(n_pixels):
            ax.plot(rf_time, rf_flat[:, i], color=colors[i], lw=1)

        # Shade baseline region (last n_baseline frames, matching the make() logic)
        ax.axvspan(rf_time[-n_baseline], rf_time[-1], color='gray', alpha=0.2,
                   label=f'baseline ({n_baseline} frames)')

        # Mark peak amplitude as horizontal reference lines
        ax.axhline(peak_amplitude, color='red', lw=0.5, ls='--', alpha=0.7)
        ax.axhline(-peak_amplitude, color='red', lw=0.5, ls='--', alpha=0.7,
                   label=f'±peak = {peak_amplitude:.3g}')

        # Noise band
        ax.axhline(noise_std, color='k', lw=0.5, ls=':', alpha=0.5)
        ax.axhline(-noise_std, color='k', lw=0.5, ls=':', alpha=0.5,
                   label=f'±noise std = {noise_std:.3g}')

        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('RF amplitude')
        ax.set_title(f"SNR = {snr:.2f}")
        ax.legend(loc='best', fontsize=8)

        return ax
