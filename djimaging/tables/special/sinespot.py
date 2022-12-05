from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.optional import ChirpQITemplate
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.math_utils import normalize_soft_zero_one


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
        suppression_index:  float   # index for suppression of larger spot
        response_rep_x_dir: longblob  # Response matrix used to compute suppression
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

    def make(self, key):
        trace, tracetimes = (self.preprocesstraces_table() & key).fetch1("preprocess_trace", "preprocess_trace_times")
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        ntrigger_rep = (self.stimulus_table() & key).fetch1('ntrigger_rep')

        response_rep_x_dir = compute_sinespot_response_matrix(trace, tracetimes, triggertimes, ntrigger_rep, delay=0.1)
        suppression_idx = compute_suppression_idx(response_rep_x_dir)

        self.insert1(dict(key, suppression_index=suppression_idx, response_rep_x_dir=response_rep_x_dir))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        response_rep_x_dir, suppression_index = (self & key).fetch1("response_rep_x_dir", "suppression_index")

        vabsmax = np.max(np.abs(response_rep_x_dir))

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(response_rep_x_dir, vmin=-vabsmax, vmax=vabsmax, cmap='coolwarm')
        plt.colorbar(im, ax=ax)
        ax.set(xlabel='Condition (Time)', ylabel='Repetition', title=f"suppression_index={suppression_index:.2f}")
        plt.tight_layout()
        plt.show()

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        suppression_index = (self & restriction).fetch("suppression_index")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hist(suppression_index)
        ax.set(title="suppression_index")
        plt.show()


def compute_sinespot_response_matrix(trace, times, triggertimes, ntrigger_rep, delay=0.1):
    from djimaging.utils.scanm_utils import split_trace_by_reps

    trace = normalize_soft_zero_one(trace, dq=5, clip=False) * 2. - 1.

    tt_reps = triggertimes.reshape(-1, ntrigger_rep)

    response_rep_x_dir = np.full(tt_reps.shape, np.nan)

    for i, tt_rep in enumerate(tt_reps):
        splits, splits_time, splits_triggertime, _ = split_trace_by_reps(
            trace=trace, times=times, triggertimes=tt_rep, ntrigger_rep=1, delay=delay, atol=0.05,
            allow_drop_last=False)
        response_rep_x_dir[i, :] = np.mean(splits[2:-2, :], axis=0)

    return response_rep_x_dir


def compute_suppression_idx(response_rep_x_dir):
    suppression_idx = np.median(response_rep_x_dir[:, 0] - response_rep_x_dir[:, 1])
    return suppression_idx
