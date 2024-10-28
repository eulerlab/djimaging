"""Fast STA table template for receptive field estimation

Example usage:
@schema
class FastStaParams(receptivefield.STAParamsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation

@schema
class FastSta(receptivefield.STATemplate):
    params_table = FastStaParams
    presentation_table = Presentation
    traces_table = PreprocessTraces
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.receptive_fields.fit_rf_utils import build_design_matrix, get_rf_timing_params, \
    compute_linear_rf_single_or_batch, compute_rf_sta
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
        stimtime : longblob  # Time of stimulus
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
            skip_duplicates=False):
        """Add default preprocess parameter to table"""

        if stim_names is None:
            stim_names = (self.stimulus_table()).fetch('stim_name')

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

            stim, stim_dict = (self.stimulus_table() & stim_key).fetch1("stim_trace", "stim_dict")
            triggertimes = (self.presentation_table() & stim_key).fetch('triggertimes')

            dt_trigger = np.median([get_mean_dt(tti, rtol_error=np.inf, rtol_warning=0.5)[0] for tti in triggertimes])
            n_trigger = int(np.percentile([tti.size for tti in triggertimes], q=95))
            triggertimes = np.arange(n_trigger) * dt_trigger

            x_stimulus, stimtime, rf_time, dims, burn_in, shift, dt, t0 = self.prepare_stimulus(
                stim, triggertimes, stim_dict.get('ntrigger_per_frame', 1), fupsample_stim,
                filter_dur_s_past, filter_dur_s_future, dtype=np.float32)

            new_key = stim_key.copy()
            new_key['x_stimulus'] = x_stimulus
            new_key['rf_time'] = rf_time
            new_key['dt'] = dt
            new_key['shift'] = shift
            new_key['stimtime'] = stimtime
            new_key['dims'] = dims
            new_key['burn_in'] = burn_in

            self.insert1(new_key, skip_duplicates=skip_duplicates)

    @staticmethod
    def prepare_stimulus(stim, triggertimes, ntrigger_per_frame, fupsample_stim, filter_dur_s_past, filter_dur_s_future,
                         dtype=np.float32):
        stimtime, stim = preprocess_stimulus(stim, triggertimes, ntrigger_per_frame, fupsample_stim)
        dt, dt_rel_error = get_mean_dt(stimtime, rtol_error=np.inf, rtol_warning=0.5)
        t0 = stimtime[0]

        rf_time, dim_t, shift, burn_in = get_rf_timing_params(filter_dur_s_past, filter_dur_s_future, dt)
        dims = (dim_t,) + stim.shape[1:]

        stim = finalize_stim(stim)
        x_stimulus = build_design_matrix(stim, n_lag=dim_t, shift=shift, dtype=dtype)

        return x_stimulus, stimtime, rf_time, dims, burn_in, shift, dt, t0


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
            return self.params_table() * self.presentation_table().proj()
        except (AttributeError, TypeError):
            pass

    def _make_compute(self, key):
        (stimtime, x_stimulus, rf_time, burn_in, shift, kind, dims, fupsample_trace, fit_kind, lowpass_cutoff,
         pre_blur_sigma_s, post_blur_sigma_s) = (self.params_table() & key).fetch1(
            'stimtime', 'x_stimulus', 'rf_time', 'burn_in', 'shift', 'fit_kind', 'dims',
            "fupsample_trace", "fit_kind", "lowpass_cutoff", "pre_blur_sigma_s", "post_blur_sigma_s")

        trace_t0s, trace_dts, traces, trace_keys = (self.traces_table() & key).fetch(
            self._traces_prefix + 'trace_t0', self._traces_prefix + 'trace_dt', self._traces_prefix + 'trace', 'KEY')

        if not np.allclose(trace_dts, trace_dts[0]):
            raise ValueError("Different trace dts not supported")
        if not np.all([trace.size == traces[0].size for trace in traces]):
            raise ValueError("Different trace sizes not supported")

        tracetime_shared = np.arange(traces[0].size) * trace_dts[0]

        rfs = []
        for trace_t0, trace in zip(trace_t0s, traces):
            tracetime, trace = preprocess_trace(
                tracetime_shared + trace_t0, trace, fupsample_trace, fit_kind, lowpass_cutoff, pre_blur_sigma_s)
            trace, dt, t0, dt_rel_error = align_trace_to_stim(stimtime=stimtime, trace=trace, tracetime=tracetime)
            trace = finalize_trace(trace, dt, post_blur_sigma_s).astype(x_stimulus.dtype)

            assert trace.size == x_stimulus.shape[0], f"stim-shape: {x_stimulus.shape}, trace-shape: {trace.shape}"
            rf = compute_rf_sta(x_stimulus, trace).reshape(dims)

            rfs.append(rf)

        return rfs, trace_keys

    def make(self, key, verbose=False, plot=False):
        rfs, trace_keys = self._make_compute(key)

        for rf, trace_key in zip(rfs, trace_keys):
            data_key = key.copy()
            data_key.update(trace_key)
            data_key['rf'] = rf.astype(np.float32)

            self.insert1(data_key)

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)

        from matplotlib import pyplot as plt

        raw_trace_t0, raw_trace_dt, raw_trace = (self.traces_table() & key).fetch1(
            self._traces_prefix + 'trace_t0', self._traces_prefix + 'trace_dt', self._traces_prefix + 'trace')

        raw_tracetime = np.arange(raw_trace.size) * raw_trace_dt + raw_trace_t0

        noise_t0, noise_dt, trace, stim_idxs = (self & key).fetch1('noise_t0', 'noise_dt', 'trace', 'stim_idxs')
        assert trace.shape[0] == stim_idxs.shape[0], "Trace and stim have different lengths"

        stim = (self.stimulus_table() & key).fetch1("stim_trace")
        stim = stim[stim_idxs]

        tracetime = np.arange(trace.size) * noise_dt + noise_t0

        fit_kind = (self.params_table() & key).fetch1('fit_kind')

        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex='all')
        ax = axs[0]
        ax.plot(tracetime, trace, label='output trace')
        ax.legend(loc='upper left')
        ax = ax.twinx()
        ax.vlines(tracetime[1:][np.any(np.diff(stim, axis=0) > 0, axis=tuple(np.arange(1, stim.ndim)))], 0, 1,
                  label='stim changes', alpha=0.1, color='k')
        ax.legend(loc='upper right')
        ax.set(xlabel='Time', title=fit_kind)
        ax.set_xlim(xlim)

        ax = axs[1]
        ax.plot(tracetime, trace, label='output trace')
        ax.legend(loc='upper left')
        ax = ax.twinx()
        ax.plot(raw_tracetime, raw_trace, 'r-', label='input trace', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set(xlabel='Time', title='Input trace')
        plt.tight_layout()
