from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.receptivefield.rf_utils import get_mean_dt
from djimaging.tables.response import ChirpQITemplate
from djimaging.utils.dj_utils import get_primary_key


class SineSpotQITemplate(ChirpQITemplate):
    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.snippets_table() & \
                (self.stimulus_table() & "stim_name = 'sinespot' or stim_family = 'sinespot'")
        except TypeError:
            pass


class SineSpotFeaturesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
        #Computes an OnOff and a transience index based on the chirp step response
        -> self.preprocesstraces_table
        ---
        surround_index : float  # large spot response - small spot response
        offset_index : float  # strongest small spot response - small spot response
        response_spot_small : float  # Mean response to small spot
        response_spot_large : float  # Mean response to large spot
        response_spot_a : float  # Mean response to small offset spot
        response_spot_b : float  # Mean response to small offset spot
        response_spot_c : float  # Mean response to small offset spot
        response_spot_d : float  # Mean response to small offset spot
        response_rep_x_cond: longblob  # Response matrix used to compute suppression
        '''
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def preprocesstraces_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.preprocesstraces_table() & \
                (self.stimulus_table() & "stim_name = 'sinespot' or stim_family = 'sinespot'")
        except TypeError:
            pass

    _delay = 0.1
    _rep_dt = 0.8

    def make(self, key):
        trace, tracetimes = (self.preprocesstraces_table() & key).fetch1("preprocess_trace", "preprocess_trace_times")
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        ntrigger_rep = (self.stimulus_table() & key).fetch1('ntrigger_rep')

        response_rep_x_cond = compute_sinespot_response_matrix(
            trace, tracetimes, triggertimes, ntrigger_rep, delay=self._delay, rep_dt=self._rep_dt)

        r_small, r_large, r_a, r_b, r_c, r_d = np.mean(response_rep_x_cond, axis=0)

        surround_index = r_large - r_small
        offset_index = np.max([r_a, r_b, r_c, r_d]) - r_small

        self.insert1(dict(key, surround_index=surround_index, offset_index=offset_index,
                          response_spot_small=r_small, response_spot_large=r_large,
                          response_spot_a=r_a, response_spot_b=r_b,
                          response_spot_c=r_c, response_spot_d=r_d,
                          response_rep_x_cond=response_rep_x_cond))

    def plot1(self, key=None, plot_trace=False):
        key = get_primary_key(table=self, key=key)

        if plot_trace:
            (self.preprocesstraces_table & key).plot1()

        response_rep_x_dir = (self & key).fetch1("response_rep_x_cond")

        vabsmax = np.max(np.abs(response_rep_x_dir))

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(response_rep_x_dir, vmin=-vabsmax, vmax=vabsmax, cmap='coolwarm')
        plt.colorbar(im, ax=ax)
        ax.set(xlabel='Condition (Time)', ylabel='Repetition')
        plt.tight_layout()
        plt.show()

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        response_spot_small = (self & restriction).fetch("response_spot_small")
        response_spot_large = (self & restriction).fetch("response_spot_large")
        surround_index = (self & restriction).fetch("surround_index")
        offset_index = (self & restriction).fetch("offset_index")

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))

        ax = axs[0]
        ax.hist(response_spot_small)
        ax.set(title="small")

        ax = axs[1]
        ax.hist(response_spot_large)
        ax.set(title="large")

        ax = axs[2]
        ax.hist(surround_index)
        ax.set(title="surround_index")

        ax = axs[3]
        ax.hist(offset_index)
        ax.set(title="offset_index")

        plt.tight_layout()
        plt.show()


def compute_sinespot_response_matrix(trace, times, triggertimes, ntrigger_rep, delay=0.1, rep_dt=0.8):
    """Split data into responses to different reps and summarize as mean response"""
    from djimaging.utils.scanm_utils import split_trace_by_reps

    dt = get_mean_dt(times)[0]
    n_frames = int(np.ceil(rep_dt / dt))

    tt_reps = triggertimes.reshape(-1, ntrigger_rep)
    response_rep_x_cond = np.full(tt_reps.shape, np.nan)

    norm_trace = trace / np.std(trace)

    for i, tt_rep in enumerate(tt_reps):
        splits, splits_time, splits_triggertime, _ = split_trace_by_reps(
            trace=norm_trace, times=times, triggertimes=tt_rep, ntrigger_rep=1, delay=delay, atol=0.05,
            allow_drop_last=False)

        response_rep_x_cond[i, :] = np.median(splits[:n_frames, :], axis=0)

    return response_rep_x_cond
