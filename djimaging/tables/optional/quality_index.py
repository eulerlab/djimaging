import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import PlaceholderTable


class QITemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

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

    stimulus_table = PlaceholderTable
    snippets_table = PlaceholderTable

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
