"""
Computes statistics of the quality index for all ROIs in one presentation.

Example usage:

from djimaging.tables import response

@schema
class ChirpQIPresentation(response.RepeatQIPresentationTemplate):
    _qidx_col = 'qidx'
    _levels = (0.25, 0.3, 0.35, 0.4, 0.45)
    presentation_table = Presentation
    qi_table = ChirpQI

@schema
class BarQIPresentation(response.RepeatQIPresentationTemplate):
    _qidx_col = 'd_qi'
    _levels = (0.5, 0.6, 0.7)
    presentation_table = Presentation
    qi_table = OsDsIndexes
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np


class RepeatQIPresentationTemplate(dj.Computed):
    database = ""
    _qidx_col = 'qidx'  # name of the column in the QI table
    _levels = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)  # levels for fraction of ROIs with qidx >= level

    @property
    def definition(self):
        definition = '''
        # Summarizes Quality Indexes for all ROIs in one presentation 
        -> self.presentation_table
        ---
        p_qidx_n_rois : int  # Number of ROIs
        p_qidx_median: float  # median quality index
        p_qidx_mean: float   # mean quality index
        p_qidx_min: float   # min quality index
        p_qidx_max: float   # max quality index
        '''
        for level in self._levels:
            definition += f'p_qidx_f{int(level * 100):03d}: float   # fraction of ROIs with qidx >= {level}\n'
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def qi_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.presentation_table().proj() & self.qi_table().proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        qidxs = (self.qi_table() & key).fetch(self._qidx_col)
        new_key = dict(
            key,
            p_qidx_n_rois=len(qidxs),
            p_qidx_mean=np.mean(qidxs),
            p_qidx_median=np.median(qidxs),
            p_qidx_min=np.min(qidxs),
            p_qidx_max=np.max(qidxs),
        )
        for level in self._levels:
            new_key[f'p_qidx_f{int(level * 100):03d}'] = np.mean(qidxs >= level)

        self.insert1(new_key)
