from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.tables.receptivefield.rf_utils import prepare_data
from djimaging.utils.dj_utils import get_primary_key


class DNoiseTraceParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        dnoise_params_id: int # unique param set id
        ---
        fit_kind : varchar(255)
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

    @property
    def definition(self):
        definition = '''
        # Preprocess traces for receptive field estimation
        -> self.traces_table
        -> self.params_table
        ---
        dt : float  # Time-step of time component
        time : longblob  # Time lof aligned traces and stimulus
        trace : longblob   # Trace to fit
        stim : longblob  # Stimulus frames
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
        trace, tracetime = (self.traces_table() & key).fetch1('preprocess_trace', 'preprocess_trace_times')
        fupsample_trace, fupsample_stim, fit_kind, lowpass_cutoff, pre_blur_sigma_s, post_blur_sigma_s, ref_time = (
                self.params_table() & key).fetch1(
            "fupsample_trace", "fupsample_stim", "fit_kind", "lowpass_cutoff",
            "pre_blur_sigma_s", "post_blur_sigma_s", "ref_time")

        stim, trace, dt, t0, dt_rel_error = prepare_data(
            trace=trace, tracetime=tracetime, stim=stim, triggertimes=triggertimes,
            ntrigger_per_frame=stim_dict.get('ntrigger_per_frame', 1),
            fupsample_trace=fupsample_trace, fupsample_stim=fupsample_stim, ref_time=ref_time,
            fit_kind=fit_kind, lowpass_cutoff=lowpass_cutoff,
            pre_blur_sigma_s=pre_blur_sigma_s, post_blur_sigma_s=post_blur_sigma_s)

        time = np.arange(trace.size) * dt + t0

        data_key = key.copy()
        data_key['dt'] = dt
        data_key['time'] = time
        data_key['trace'] = trace
        data_key['stim'] = stim
        data_key['dt_rel_error'] = dt_rel_error
        self.insert1(data_key)

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)

        from matplotlib import pyplot as plt

        raw_trace, raw_tracetime = (self.traces_table() & key).fetch1('preprocess_trace', 'preprocess_trace_times')

        time, trace, stim = (self & key).fetch1('time', 'trace', 'stim')
        assert time.shape[0] == trace.shape[0], (time.shape[0], trace.shape[0])
        assert time.shape[0] == stim.shape[0], (time.shape[0], stim.shape[0])

        fit_kind = (self.params_table() & key).fetch1('fit_kind')

        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex='all')
        ax = axs[0]
        ax.plot(time, trace, label='output trace')
        ax.legend(loc='upper left')
        ax = ax.twinx()
        ax.vlines(time[1:][np.any(np.diff(stim, axis=0) > 0, axis=tuple(np.arange(1, stim.ndim)))], 0, 1,
                  label='stim changes', alpha=0.1, color='k')
        ax.legend(loc='upper right')
        ax.set(xlabel='Time', title=fit_kind)
        ax.set_xlim(xlim)

        ax = axs[1]
        ax.plot(time, trace, label='output trace')
        ax.legend(loc='upper left')
        ax = ax.twinx()
        ax.plot(raw_tracetime, raw_trace, 'r-', label='input trace', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set(xlabel='Time', title='Input trace')
        plt.tight_layout()
