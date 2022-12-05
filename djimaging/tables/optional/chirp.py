from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_trace_and_trigger


class ChirpQITemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = f'''
        #Computes the QI index for (chirp) responses as described in Baden et al. (2016) for chirp
        -> self.snippets_table
        ---
        qidx: float   # chirp quality index
        min_qidx: float   # minimum quality index as 1/r (r = #repetitions)
        '''
        return definition

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
            return self.snippets_table() & (self.stimulus_table() & "stim_name = 'chirp' or stim_family = 'chirp'")
        except TypeError:
            pass

    def make(self, key):
        snippets = (self.snippets_table() & key).fetch1('snippets')
        assert snippets.ndim == 2
        qidx = np.var(np.mean(snippets, axis=1)) / np.mean(np.var(snippets, axis=0))
        min_qidx = 1 / snippets.shape[1]
        self.insert1(dict(key, qidx=qidx, min_qidx=min_qidx))

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        qidx, min_qidx = (self & restriction).fetch('qidx', 'min_qidx')
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        ax = axs[0]
        ax.set(title='qidx')
        ax.hist(qidx)

        ax = axs[1]
        ax.set(title='qidx > min_qidx')
        ax.hist(np.array(qidx > min_qidx).astype(int))

        plt.tight_layout()
        plt.show()


class ChirpFeaturesTemplate(dj.Computed):
    # TODO: Add more features

    database = ""

    @property
    def definition(self):
        definition = '''
        #Computes an OnOff and a transience index based on the chirp step response
        -> self.snippets_table
        ---
        on_off_index:       float   # index indicating light preference (-1 Off, 1 On)
        transience_index:   float   # index indicating transience of response
        '''
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.snippets_table() & (self.stimulus_table() & "stim_name = 'chirp' or stim_family = 'chirp'")
        except TypeError:
            pass

    def make(self, key):
        # TODO: Should this depend on pres? Triggertimes are also in snippets and sf can be derived from times
        snippets, snippets_times = (self.snippets_table() & key).fetch1('snippets', 'snippets_times')
        trigger_times = (self.presentation_table() & key).fetch1('triggertimes')
        sf = (self.presentation_table().ScanInfo() & key).fetch1('scan_frequency')

        on_off_index = compute_on_off_index(snippets, snippets_times, trigger_times, sf)
        transience_index = compute_transience_index(snippets, snippets_times, trigger_times, sf)

        self.insert1(dict(key, on_off_index=on_off_index, transience_index=transience_index))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        snippets, snippets_times, triggertimes_snippets = (self.snippets_table() & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        on_off_index, transience_index = (self & key).fetch1('on_off_index', 'transience_index')

        ax = plot_trace_and_trigger(
            time=snippets_times, trace=snippets, triggertimes=triggertimes_snippets, title=str(key))

        ax.set(title=f"on_off_index={on_off_index:.2f}\ntransience_index={transience_index:.2f}")

        plt.tight_layout()
        plt.show()

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        on_off_index, transience_index = (self & restriction).fetch('on_off_index', 'transience_index')

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        ax = axs[0]
        ax.set(title='on_off_index')
        ax.hist(on_off_index)

        ax = axs[1]
        ax.set(title='transience_index')
        ax.hist(transience_index)

        plt.tight_layout()
        plt.show()


def compute_on_off_index(snippets, snippets_times, trigger_times, sf, light_step_duration=1):
    start_trigs = trigger_times[::2]
    light_step_frames = int(light_step_duration * sf)

    on_responses = np.zeros((light_step_frames, snippets.shape[1]))
    off_responses = np.zeros((light_step_frames, snippets.shape[1]))

    for i in range(snippets.shape[1]):
        step_up = start_trigs[i] + 2
        step_down = start_trigs[i] + 5

        snip_times = snippets_times[:, i]
        snip = snippets[:, i]
        snip = snip - snip.min()

        step_up_idx = np.argmax(np.isclose(snip_times, step_up, atol=1e-01))
        step_down_idx = np.argmax(np.isclose(snip_times, step_down, atol=1e-01))

        baseline_on = np.median(snip[:step_up_idx])
        on_response = snip[step_up_idx:step_up_idx + light_step_frames] - baseline_on
        on_response[on_response < 0] = 0

        baseline_off = np.median(snip[step_down_idx - light_step_frames:step_down_idx])
        off_response = snip[step_down_idx:step_down_idx + light_step_frames] - baseline_off
        off_response[off_response < 0] = 0

        on_responses[:, i] = on_response
        off_responses[:, i] = off_response

    avg_on = on_responses[4:, :].mean()
    avg_off = off_responses[4:, :].mean()

    if avg_on + avg_off < 1e-10:
        on_off_index = 0
    else:
        on_off_index = (avg_on - avg_off) / (avg_on + avg_off)

    return on_off_index


def compute_transience_index(snippets, snippets_times, trigger_times, sf, upsam_fre=500):
    upsampling_factor = int(snippets.shape[0] / sf * upsam_fre)

    resampled_snippets = np.zeros((upsampling_factor, snippets.shape[1]))
    resampled_snippets_times = np.zeros((upsampling_factor, snippets.shape[1]))

    for i in range(snippets.shape[1]):
        resampled_snippets[:, i] = signal.resample(snippets[:, i], int(upsampling_factor))

        start_time = snippets_times[0, i]
        end_time = snippets_times[-1, i]
        resampled_snippets_times[:, i] = np.linspace(start_time, end_time, upsampling_factor)
        # used this as interpolartion, signal.resample had edge effects when interpolaring the linear array

    start_trigs_local = trigger_times[::2]

    transience_indexes = np.zeros(snippets.shape[1])
    for i in range(snippets.shape[1]):
        stim_start = start_trigs_local[i]
        stim_end = start_trigs_local[i] + 6
        snip_times = resampled_snippets_times[:, i]
        snip = resampled_snippets[:, i]
        snip = snip - snip.min()
        stim_start_idx = np.argmax(np.isclose(snip_times, stim_start, atol=1e-01))
        stim_end_idx = np.argmax(np.isclose(snip_times, stim_end, atol=1e-01))
        trace = snip[stim_start_idx:stim_end_idx]
        peak = np.argmax(trace)
        peak_alpha = np.argmax(np.isclose(snip_times, snip_times[peak] + 0.4, atol=1e-01))
        transience_indexes[i] = 1 - (snip[stim_start_idx + peak_alpha] / snip[stim_start_idx + peak])

    transience_index = np.mean(transience_indexes)
    return transience_index
