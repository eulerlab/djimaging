from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt


class RepeatQITemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = f'''
        # Computes the QI index for responses for repeated stimuli as a signal to noise ratio
        -> self.snippets_table
        ---
        qidx: float   # quality index as signal to noise ratio
        min_qidx: float   # minimum quality index as 1/r (r = #repetitions)
        '''
        return definition

    @property
    @abstractmethod
    def _stim_family(self):
        return None

    @property
    @abstractmethod
    def _stim_name(self):
        return None

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    def get_stim_restriction(self):
        if (self._stim_family is not None) and (self._stim_name is not None):
            return f"stim_family='{self._stim_family}' or stim_name='{self._stim_name}'"
        elif self._stim_family is not None:
            return f"stim_family='{self._stim_family}'"
        elif self._stim_name is not None:
            return f"stim_name='{self._stim_name}'"
        else:
            return dict()

    @property
    def key_source(self):
        try:
            return self.snippets_table().proj() & \
                (self.stimulus_table() & "isrepeated=1" & self.get_stim_restriction())
        except (AttributeError, TypeError):
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


class RepeatQIPresentationTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = f'''
        # Summarizes Quality Indexes for all ROIs in one presentation 
        -> self.presentation_table
        -> self.qi_table
        ---
        mean_qidx: float   # mean quality index
        min_qidx: float   # mean quality index
        max_qidx: float   # mean quality index
        '''
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def qi_table(self):
        pass
