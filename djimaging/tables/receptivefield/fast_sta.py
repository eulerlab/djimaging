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
            self, stim_names=None, dnoise_params_id=1, fit_kind="gradient",
            fupsample_trace=10, fupsample_stim=10, lowpass_cutoff=0,
            pre_blur_sigma_s=0, post_blur_sigma_s=0,
            filter_dur_s_past=1., filter_dur_s_future=0.,
            skip_duplicates=False, verbose=True):
        """Add default preprocess parameter to table"""

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

            if framerate > 0:
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
                      f"n_trigger: {n_trigger}, dt_trigger: {dt_trigger}")

            x_stimulus, rf_time, dims, burn_in, shift, dt, t0 = self.prepare_stimulus(
                stim, stimtime, stim_dict.get('ntrigger_per_frame', 1), fupsample_stim,
                filter_dur_s_past, filter_dur_s_future)

            new_key = stim_key.copy()
            new_key['x_stimulus'] = x_stimulus
            new_key['rf_time'] = rf_time
            new_key['dt'] = dt
            new_key['shift'] = shift
            new_key['dims'] = dims
            new_key['burn_in'] = burn_in

            self.insert1(new_key, skip_duplicates=skip_duplicates)

    @staticmethod
    def prepare_stimulus(stim, triggertimes, ntrigger_per_frame, fupsample_stim, filter_dur_s_past,
                         filter_dur_s_future):
        stimtime, stim = preprocess_stimulus(stim, triggertimes, ntrigger_per_frame, fupsample_stim)
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

    def populate(
            self,
            *restrictions,
            suppress_errors=False,
            return_exception_objects=False,
            reserve_jobs=False,
            order="original",
            limit=None,
            max_calls=None,
            display_progress=False,
            processes=1,
            make_kwargs=None,
    ):
        restrictions = [{}] if len(restrictions) == 0 else restrictions
        if len(self.params_table() & [*restrictions]) != 1:
            raise NotImplementedError("Only one parameter set supported. Provide `restrictions` that selects one.")

        (x_stimulus, dt, rf_time, burn_in, shift, kind, dims, fupsample_stim, fupsample_trace,
         fit_kind, lowpass_cutoff, pre_blur_sigma_s, post_blur_sigma_s) = (
                self.params_table & [*restrictions]).fetch1(
            'x_stimulus', 'dt', 'rf_time', 'burn_in', 'shift', 'fit_kind', 'dims', 'fupsample_stim',
            "fupsample_trace", "fit_kind", "lowpass_cutoff", "pre_blur_sigma_s", "post_blur_sigma_s")

        if make_kwargs is None:
            make_kwargs = dict()

        x_stimulus = np.ascontiguousarray(x_stimulus.astype(np.float32))

        make_kwargs['sta_params'] = dict(
            x_stimulus=x_stimulus, rf_time=rf_time, burn_in=burn_in, shift=shift, kind=kind, dims=dims,
            dt=dt, fupsample_trace=fupsample_trace, fupsample_stim=fupsample_stim,
            fit_kind=fit_kind, lowpass_cutoff=lowpass_cutoff,
            pre_blur_sigma_s=pre_blur_sigma_s, post_blur_sigma_s=post_blur_sigma_s,
        )

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

    def make(self, key, verbose=False, plot=False, sta_params=None):
        if isinstance(sta_params, dict):
            x_stimulus = sta_params['x_stimulus']
            dims = sta_params['dims']
            fupsample_trace = sta_params['fupsample_trace']
            fupsample_stim = sta_params['fupsample_stim']
            fit_kind = sta_params['fit_kind']
            lowpass_cutoff = sta_params['lowpass_cutoff']
            pre_blur_sigma_s = sta_params['pre_blur_sigma_s']
            post_blur_sigma_s = sta_params['post_blur_sigma_s']
        else:
            raise NotImplementedError("Only dict supported")

        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')

        dt_triggertimes = np.median(np.diff(triggertimes))
        xidxs = np.arange(triggertimes.size + 1)
        xidxs_stim = np.repeat(xidxs, fupsample_stim) + np.tile(
            np.arange(fupsample_stim) * 1 / fupsample_stim, xidxs.size)

        stimtime = np.interp(
            xidxs_stim, xidxs, np.append(triggertimes, triggertimes[-1] + dt_triggertimes))[:-fupsample_stim]

        trace_t0s, trace_dts, traces, rf_keys = (self.traces_table() & key).fetch(
            self._traces_prefix + 'trace_t0', self._traces_prefix + 'trace_dt', self._traces_prefix + 'trace', 'KEY')

        if len(rf_keys) == 0:
            print(f"No traces found for {key}")
            return

        tracetime_base = np.arange(traces[0].size) * trace_dts[0]

        rf_entries = []
        for trace, trace_t0, rf_key in tqdm(
                zip(traces, trace_t0s, rf_keys), total=len(rf_keys), desc=str(key), leave=False):
            rf = self._compute_rf(
                tracetime_base + trace_t0, trace, stimtime, x_stimulus,
                dims, fupsample_trace, fit_kind, lowpass_cutoff, pre_blur_sigma_s, post_blur_sigma_s)
            rf_entries.append({**key, **rf_key, "rf": rf})

        self.insert(rf_entries)

    @staticmethod
    def _compute_rf(tracetime, trace, stimtime, x_stimulus,
                    dims, fupsample_trace, fit_kind, lowpass_cutoff, pre_blur_sigma_s,
                    post_blur_sigma_s):

        tracetime, trace = preprocess_trace(
            tracetime, trace, fupsample_trace, fit_kind, lowpass_cutoff, pre_blur_sigma_s)
        trace, dt, t0, dt_rel_error = align_trace_to_stim(
            stimtime=stimtime, trace=trace, tracetime=tracetime)
        trace = finalize_trace(trace, dt, post_blur_sigma_s).astype(np.float32)

        assert trace.size <= x_stimulus.shape[0], f"stim-shape: {x_stimulus.shape}, trace-shape: {trace.shape}"
        assert trace.dtype == x_stimulus.dtype, (trace.dtype, x_stimulus.dtype)
        rf = compute_rf_sta(np.ascontiguousarray(x_stimulus[:trace.size]),
                            np.ascontiguousarray(trace)).reshape(dims)

        return rf

    def plot1(self, key=None, downsample=1):
        self.plot1_frames(key=key, downsample=downsample)

    def plot1_frames(self, key=None, downsample=1):
        key = get_primary_key(table=self, key=key)
        rf_time = (self.params_table & key).fetch1('rf_time')
        rf = (self & key).fetch1('rf')
        plot_rf_frames(rf, rf_time, downsample=downsample)

    def plot1_video(self, key=None, fps=10):
        key = get_primary_key(table=self, key=key)
        rf_time = (self.params_table & key).fetch1('rf_time')
        rf = (self & key).fetch1('rf')
        return plot_rf_video(rf, rf_time, fps=fps)
