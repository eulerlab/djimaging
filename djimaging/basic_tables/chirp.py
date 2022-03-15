import datajoint as dj
import numpy as np
from scipy import signal

from djimaging.utils.dj_utils import PlaceholderTable


class ChirpQITemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = '''
        #Computes the QI index for chirp responses as described in Baden et al. (2016)
        -> self.detrendsnippets_table
        ---
        chirp_qi:   float   # chirp quality index
        min_qi:     float   # minimum quality index as 1/r (r = #repetitions)
        '''
        return definition

    detrendsnippets_table = PlaceholderTable

    @property
    def key_source(self):
        return self.detrendsnippets_table() & "stim_id=1"

    def make(self, key):
        snippets = (self.detrendsnippets_table() & key).fetch1('detrend_snippets')
        assert snippets.ndim == 2
        chirp_qi = np.var(np.mean(snippets, axis=1)) / np.mean(np.var(snippets, axis=0))
        min_qi = 1 / snippets.shape[1]
        self.insert1(dict(key, chirp_qi=chirp_qi, min_qi=min_qi))


class ChirpFeaturesTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = '''
        #Computes an OnOff and a transience index based on the chirp step response
        -> self.detrendsnippets_table
        -> self.presentation_table
        ---
        on_off_index:       float   # index indicating light preference (-1 Off, 1 On)
        transience_index:   float   # index indicating transience of response
        '''
        return definition

    detrendsnippets_table = PlaceholderTable
    presentation_table = PlaceholderTable

    @property
    def key_source(self):
        return self.detrendsnippets_table() & "stim_id=1"

    def make(self, key):
        # TODO: Make this code readable
        # TODO: Should this depend on pres? Triggertimes are also in snippets and sf can be derived from times
        snippets = (self.detrendsnippets_table() & key).fetch1('detrend_snippets')
        snippets_times = (self.detrendsnippets_table() & key).fetch1('detrend_snippets_times')
        trigger_times = (self.presentation_table() & key).fetch1('triggertimes')
        sf = (self.presentation_table() & key).fetch1('scan_frequency')
        start_trigs = trigger_times[::2]
        light_step_duration = 1
        light_step_frames = int(light_step_duration * sf)

        # On-Off Index
        on_responses = np.zeros((light_step_frames, snippets.shape[1]))
        off_responses = np.zeros((light_step_frames, snippets.shape[1]))
        for i in range(snippets.shape[1]):
            step_up = start_trigs[i] + 2
            snip_times = snippets_times[:, i]
            snip = snippets[:, i]
            snip = snip - snip.min()
            step_up_idx = \
                np.nonzero(np.isclose(snip_times,
                                      np.ones_like(snip_times) * step_up,
                                      atol=1e-01))[0][0]
            step_down = start_trigs[i] + 5
            step_down_idx = \
                np.nonzero(np.isclose(snip_times,
                                      np.ones_like(snip_times) * step_down,
                                      atol=1e-01))[0][0]
            baseline_on = np.median(snip[:step_up_idx])
            on_response = snip[step_up_idx:step_up_idx + light_step_frames]
            off_response = \
                snip[step_down_idx:step_down_idx + light_step_frames]
            on_response = on_response - baseline_on
            on_response[on_response < 0] = 0
            baseline_off = np.median(snip[step_down_idx - light_step_frames:step_down_idx])
            off_response = off_response - baseline_off
            off_response[off_response < 0] = 0
            on_responses[:, i] = on_response
            off_responses[:, i] = off_response
        avg_on = on_responses[4:, :].mean()
        avg_off = off_responses[4:, :].mean()
        on_off_index = (avg_on - avg_off) / (avg_on + avg_off)
        if np.isnan(on_off_index):
            on_off_index = 0
        # Transience Index
        upsam_fre = 500
        upsampling_factor = int(snippets.shape[0] / sf * upsam_fre)

        resampled_snippets = np.zeros((upsampling_factor, snippets.shape[1]))
        resampled_snippets_times = np.zeros((upsampling_factor, snippets.shape[1]))

        for i in range(snippets.shape[1]):
            resampled_snippets[:, i] = signal.resample(snippets[:, i], int(upsampling_factor))

            start_time = snippets_times[0, i]
            end_time = snippets_times[-1, i]
            resampled_snippets_times[:, i] = np.linspace(start_time, end_time, upsampling_factor)
            # used this as interpolartion, signal.resample had edge effects when interpolaring the linear array

        # Local Chirp
        start_trigs_local = trigger_times[::2]

        RTi = np.zeros(snippets.shape[1])
        for i in range(snippets.shape[1]):
            stim_start = start_trigs_local[i]
            stim_end = start_trigs_local[i] + 6
            snip_times = resampled_snippets_times[:, i]
            snip = resampled_snippets[:, i]
            snip = snip - snip.min()
            stim_start_idx = np.nonzero(np.isclose(snip_times, np.ones_like(snip_times) * stim_start, atol=1e-01))[0][0]
            stim_end_idx = np.nonzero(np.isclose(snip_times, np.ones_like(snip_times) * stim_end, atol=1e-01))[0][0]
            trace = snip[stim_start_idx:stim_end_idx]
            peak = np.where(trace == trace.max())[0][0]
            peak_alpha = np.nonzero(np.isclose(snip_times, snip_times[peak] + 0.4, atol=1e-01))[0][0]
            RTi[i] = 1 - (snip[stim_start_idx + peak_alpha] / snip[stim_start_idx + peak])
        RTi = RTi.mean()

        self.insert1(dict(key, on_off_index=on_off_index, transience_index=RTi))
