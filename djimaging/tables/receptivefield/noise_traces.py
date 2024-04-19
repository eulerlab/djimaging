"""
This module contains tables for preprocessing traces for receptive field estimation.
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.receptive_fields.preprocess_rf_utils import prepare_noise_data
from djimaging.utils.dj_utils import get_primary_key


class DNoiseTraceParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        dnoise_params_id: tinyint unsigned # unique param set id
        ---
        fit_kind : varchar(191)
        fupsample_trace : tinyint unsigned  # Multiplier of sampling frequency, using linear interpolation.
        fupsample_stim = 0 : tinyint unsigned  # Multiplier of sampling stimulus, using repeat.
        lowpass_cutoff = 0: float  # Cutoff frequency low pass filter, applied if larger 0.
        pre_blur_sigma_s = 0: float  # Gaussian blur applied after low pass filter.
        post_blur_sigma_s = 0: float  # Gaussian blur applied after all other steps.
        ref_time ='trace' : enum('trace', 'stim')  # Which time to use as reference.
        """
        return definition

    def add_default(
            self, dnoise_params_id=1, fit_kind="events", ref_time='trace',
            fupsample_trace=1, fupsample_stim=1, lowpass_cutoff=0,
            pre_blur_sigma_s=0, post_blur_sigma_s=0, skip_duplicates=False):
        """Add default preprocess parameter to table"""

        key = dict(dnoise_params_id=dnoise_params_id, fit_kind=fit_kind,
                   fupsample_trace=fupsample_trace, fupsample_stim=fupsample_stim,
                   ref_time=ref_time, lowpass_cutoff=lowpass_cutoff,
                   pre_blur_sigma_s=pre_blur_sigma_s, post_blur_sigma_s=post_blur_sigma_s)

        self.insert1(key, skip_duplicates=skip_duplicates)


class DNoiseTraceTemplate(dj.Computed):
    database = ""
    _traces_prefix = 'pp_'

    @property
    def definition(self):
        definition = '''
        # Preprocess traces for receptive field estimation
        -> self.traces_table
        -> self.params_table
        ---
        trace : longblob   # Trace to fit
        stim : longblob  # Stimulus frames
        noise_dt : float  # Time-step of time component
        noise_t0 : float  # Time of first sample
        dt_rel_error : float  # Maximum relative error of dts, if too large, can have unwanted effects
        '''
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
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
    @abstractmethod
    def params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.params_table() * self.traces_table().proj() & \
                (self.stimulus_table() & "stim_family = 'noise'")
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        stim, stim_dict = (self.stimulus_table() & key).fetch1("stim_trace", "stim_dict")
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        trace_t0, trace_dt, trace = (self.traces_table() & key).fetch1(
            self._traces_prefix + 'trace_t0', self._traces_prefix + 'trace_dt', self._traces_prefix + 'trace')
        fupsample_trace, fupsample_stim, fit_kind, lowpass_cutoff, pre_blur_sigma_s, post_blur_sigma_s, ref_time = (
                self.params_table() & key).fetch1(
            "fupsample_trace", "fupsample_stim", "fit_kind", "lowpass_cutoff",
            "pre_blur_sigma_s", "post_blur_sigma_s", "ref_time")

        tracetime = np.arange(trace.size) * trace_dt + trace_t0

        stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
            trace=trace, tracetime=tracetime, stim=stim, triggertimes=triggertimes,
            ntrigger_per_frame=stim_dict.get('ntrigger_per_frame', 1),
            fupsample_trace=fupsample_trace, fupsample_stim=fupsample_stim, ref_time=ref_time,
            fit_kind=fit_kind, lowpass_cutoff=lowpass_cutoff,
            pre_blur_sigma_s=pre_blur_sigma_s, post_blur_sigma_s=post_blur_sigma_s)

        data_key = key.copy()
        data_key['stim'] = stim
        data_key['trace'] = trace
        data_key['noise_t0'] = t0
        data_key['noise_dt'] = dt
        data_key['dt_rel_error'] = dt_rel_error
        self.insert1(data_key)

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)

        from matplotlib import pyplot as plt

        raw_trace_t0, raw_trace_dt, raw_trace = (self.traces_table() & key).fetch1(
            self._traces_prefix + 'trace_t0', self._traces_prefix + 'trace_dt', self._traces_prefix + 'trace')

        raw_tracetime = np.arange(raw_trace.size) * raw_trace_dt + raw_trace_t0

        noise_t0, noise_dt, trace, stim = (self & key).fetch1('noise_t0', 'noise_dt', 'trace', 'stim')
        assert trace.shape[0].shape[0] == stim.shape[0], "Trace and stim have different lengths"

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
