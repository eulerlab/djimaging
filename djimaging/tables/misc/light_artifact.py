"""
Table for Light Artifact

Example usage:

from djimaging.tables import misc

@schema
class LightArtifact(misc.LightArtifactTemplate):
    userinfo_table = UserInfo
    presentation_table = Presentation
    raw_params_table = RawDataParams
    stimulus_table = Stimulus
"""
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.autorois.corr_roi_mask_utils import stack_corr_image
from djimaging.tables.core.averages import compute_upsampled_average
from djimaging.tables.core.snippets import get_aligned_snippets_times
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.math_utils import normalize
from djimaging.utils.scanm import read_utils
from djimaging.utils.snippet_utils import split_trace_by_reps


class LightArtifactTemplate(dj.Computed):
    database = ""
    _use_main_channel = True
    _f_resample = 60.  # Hz

    @property
    def definition(self):
        definition = """
        -> self.presentation_table
        ---
        light_artifact : longblob  # Normalized mean light artifact trace in line precision
        triggertimes_rel : blob # Relative triggertimes
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.presentation_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def make(self, key):
        filepath = (self.presentation_table & key).fetch1('pres_data_file')
        from_raw_data = (self.raw_params_table & key).fetch1('from_raw_data')
        data_name = (self.userinfo_table & key).fetch1('data_stack_name'
                                                       if self._use_main_channel else 'alt_stack_name')

        stack = read_utils.load_stacks(filepath, from_raw_data, ch_names=(data_name,))[0][data_name]
        light_artifact = stack[0, :, :].T.flatten()

        ntrigger_rep = (self.stimulus_table() & key).fetch1('ntrigger_rep')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        line_duration = (self.presentation_table.ScanInfo & key).fetch('line_duration')

        times = np.arange(light_artifact.size) * line_duration

        snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_reps(
            light_artifact, times, triggertimes, ntrigger_rep, allow_drop_last=True, pad_trace=False)

        triggertimes_rel = np.mean(triggertimes_snippets - triggertimes_snippets[0, :], axis=1)
        average = np.mean(snippets, axis=1)
        average = normalize(average, norm_kind='zero_one')

        self.insert1(dict(key, light_artifact=average, triggertimes_rel=triggertimes_rel))

    def plot1(self, key=None):
        key = get_primary_key(self, key=key)
        light_artifact = (self & key).fetch1('light_artifact')
        line_duration = (self.presentation_table.ScanInfo & key).fetch('line_duration')
        triggertimes_rel = (self & key).fetch1('triggertimes_rel')

        time = np.arange(light_artifact.size) * line_duration

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(time, light_artifact)

        ax.vlines(triggertimes_rel, ymin=np.min(light_artifact), ymax=np.max(light_artifact), colors='r',
                  linestyles='--')

        plt.show()
