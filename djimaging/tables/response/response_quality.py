from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt


class RepeatQITemplate(dj.Computed):
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

    _stim_family = None
    _stim_name = None

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
        stim_restriction = "isrepeated=1"
        if self._stim_family is not None:
            stim_restriction += f"and (stim_family='{self._stim_family}'"
        if self._stim_name is not None:
            stim_restriction += f"or stim_name='{self._stim_name}'"
        stim_restriction += ")"

        try:
            return self.snippets_table() & (self.stimulus_table() & stim_restriction)
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
