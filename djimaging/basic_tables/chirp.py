import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import PlaceholderTable


class ChirpQITemplate(dj.Computed):
    definition = '''
    #Computes the QI index for chirp responses as described in Baden et al. (2016)
    -> self.detrendsnippets_table
    ---
    chirp_qi:   float   # chirp quality index
    min_qi:     float   # minimum quality index as 1/r (r = #repetitions)
    '''

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

