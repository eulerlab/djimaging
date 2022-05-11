import datajoint as dj
import numpy as np
from scipy import stats
import cmath
import matplotlib.pyplot as plt

from djimaging.utils.dj_utils import PlaceholderTable

class OsDsIndexesV21Template(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        #This class computes the direction and orientation selectivity indexes 
        #as well as a quality index of DS responses as described in Baden et al. (2016)
        -> self.snippets_table
        ---
        ds_index:   float   #direction selectivity index as resulting vector length (absolute of projection on complex exponential)
        ds_pvalue:  float   #p-value indicating the percentile of the vector length in null distribution
        ds_null:    longblob    #null distribution of DSIs
        pref_dir:  float    #preferred direction
        os_index:   float   #orientation selectivity index in analogy to ds_index
        os_pvalue:  float   #analogous to ds_pvalue for orientation tuning
        os_null:    longblob    #null distribution of OSIs
        pref_or:    float   #preferred orientation
        d_qi:       float   #quality index for moving bar response
        time_component: longblob    #time component
        dir_component:  longblob    #direction component
        surrogate_v:    longblob    #computed by projecting on time
        surrogate_dsi:  float   #DSI of surrogate v 
        avg_sorted_resp:    longblob    # response matrix, averaged across reps
        """
        return definition

    stimulus_table = PlaceholderTable
    snippets_table = PlaceholderTable

    @property
    def key_source(self):
        return self.snippets_table() & (self.stimulus_table() & "stim_name = 'movingbar' or stim_family = 'movingbar'")

    def make(self, key):
        #if (Stimulus() & key).fetch1('stimulusname') == "ds":
        dir_order = (self.stimulus_table() & key).fetch1('trial_info')
        snippets = (self.snippets_table() & key).fetch1('snippets')  # get the response snippets

        dir_order = np.tile(dir_order, 3)
        #dir_order = Stimulus().DsInfo().fetch1('trialinfo')
        dir_deg = list(dir_order[:8])   # get the directions of the bars in degree
        dir_rad = np.deg2rad(dir_deg)   # convert to radians
        #snippets = (DetrendSnippets() & key).fetch1('detrend_snippets')  # get the response snippets

        valid = False   # check whether there are 2 or 3 complete repetitions of the stimulus sequence
        if snippets.shape[-1] == len(dir_order):
            dir_idx = [list(np.nonzero(dir_order == d)[0])
                        for d in dir_deg]
            valid = True
        elif snippets.shape[-1] == 16:
            dir_idx = [list(np.nonzero(dir_order[:-8] == d)[0])
                        for d in dir_deg]
            valid = True
        if valid:
            sorted_responses, sorted_directions = \
                self.sort_response_matrix(snippets, dir_idx, dir_rad)
            avg_sorted_responses = np.mean(sorted_responses, axis=-1)
            u, v, s = self.get_time_dir_kernels(avg_sorted_responses)
            dsi, pref_dir = self.get_si(v, sorted_directions, 1)
            osi, pref_or = self.get_si(v, sorted_directions, 2)
            (t, d, r) = sorted_responses.shape
            temp = np.reshape(sorted_responses, (t, d*r))
            projected = np.dot(np.transpose(temp), u)  # we do this whole projection thing to make the result
            projected = np.reshape(projected, (d, r))  #  between the original and the shuffled comparable
            surrogate_v = np.mean(projected, axis = -1)
            surrogate_v -= np.min(surrogate_v)
            surrogate_v /= np.max(surrogate_v)
            dsi_s, pref_dir_s = self.get_si(surrogate_v,
                                            sorted_directions,
                                            1)
            osi_s, pref_or_s = self.get_si(surrogate_v,
                                            sorted_directions,
                                            2)
            null_dist_dsi, p_dsi = self.test_tuning(np.transpose(projected),
                                                sorted_directions,
                                                1)
            null_dist_osi, p_osi = self.test_tuning(np.transpose(projected),
                                                sorted_directions,
                                                2)
            d_qi = self.quality_index_ds(sorted_responses)
            self.insert1(dict(key,
                                ds_index=dsi, ds_pvalue=p_dsi,
                                ds_null=null_dist_dsi, pref_dir=pref_dir,
                                os_index=osi, os_pvalue=p_osi,
                                os_null=null_dist_osi, pref_or=pref_or,
                                d_qi=d_qi, time_component=u, dir_component=v,
                                surrogate_v=surrogate_v, surrogate_dsi=dsi_s,
                                avg_sorted_resp=avg_sorted_responses))
            print("Populated", key)
        else:
            print("Skipped for ", key)

    def quality_index_ds(self, raw_sorted_resp_mat):
        """
        This function computes the quality index for responses to moving bar as described in
        Baden et al 2016. QI is computed for each direction separately and the best QI is taken
        Inputs:
        raw_sorted_resp_mat:    3d array (time x directions x reps per direction)
        Output:
        qi: float               quality index
        """

        n_dirs = raw_sorted_resp_mat.shape[1]
        qis = []
        for d in range(n_dirs):
            numerator = np.var(np.mean(raw_sorted_resp_mat[:, d, :], axis=-1), axis=0)
            denom = np.mean(np.var(raw_sorted_resp_mat[:, d, :], axis=0), axis=-1)
            qis.append(numerator / denom)
        return np.max(qis)

    def sort_response_matrix(self, snippets, idxs, directions):
        """
        Sorts the snippets according to stimulus condition and repetition into a time x direction x repetition matrix
        Inputs:
        snippets    list or array, time x (directions*repetitions)
        idxs        list of lists giving idxs into last axis of snippets. idxs[0] gives the indexes of rows in snippets
                    which are responses to the direction directions[0]

        Outputs:
        sorted_responses   array, time x direction x repetitions, with directions sorted(!) (0, 45, 90, ..., 315) degrees
        sorted_directions   array, sorted directions
        """
        structured_responses = snippets[:, idxs]
        sorting = np.argsort(directions)
        sorted_responses = structured_responses[:, sorting, :]
        sorted_directions = directions[sorting]
        return sorted_responses, sorted_directions

    def get_time_dir_kernels(self, sorted_responses):
        """
        Performs singular value decomposition on the time x direction matrix (averaged across repetitions)
        Uses a heuristic to try to determine whether a sign flip occurred during svd
        For the time course, the mean of the first second is subtracted and then the vector is divided by the maximum
        absolute value.
        For the direction/orientation tuning curve, the vector is normalized to the range (0,1)
        Input:
        sorted_responses:   array, time x direction

        Outputs:
        time_kernel     array, time x 1 (time component, 1st component of U)
        direction_tuning    array, directions x 1 (direction tuning, 1st component of V)
        singular_value  float, 1st singular value
        """

        U, S, V = np.linalg.svd(sorted_responses)
        u = U[:, 0]
        s = S[0]
        v = V[0, :]
        # the time_kernel determined by SVD should be correlated to the average response across all directions. if the
        # correlation is negative, U is likely flipped
        r, _ = stats.spearmanr(u, np.mean(sorted_responses, axis=-1), axis=1)
        su = np.sign(r)
        if su == 0:
            su = 1
        sv = np.sign(np.mean(np.sign(V[0, :])))
        if sv == 1 and su == 1:
            s = 1
        elif sv == -1 and su == -1:
            s = -1
        elif sv == 1 and su == -1:
            s = 1
        elif sv == 0:
            s = su
        else:
            s = 1

        u = s*u
        #determine which entries correspond to the first second, assuming 4 seconds presentation time
        idx = int(len(u)/4)
        u -= np.mean(u[:idx])
        u = u/np.max(abs(u))
        v = s*v
        v -= np.min(v)
        v /= np.max(abs(v))

        return u, v, s

    def get_si(self, v, dirs, per):
        """
        Computes direction/orientation selectivity index and preferred direction/orientation of a cell by projecting the tuning curve v on a
        complex exponential of the according directions dirs (as in Baden et al. 2016)
        Inputs:
        v:  array, dirs x 1, tuning curve as returned by SVD
        dirs:   array, dirs x 1, directions in radians
        per:    int (1 or 2), indicating whether direction (1) or orientation (2) shall be tested

        Output:
        index:  float, D/O si
        direction:  float, preferred D/O
        """
        bin_spacing = np.diff(per*dirs)[0]
        correction_factor = bin_spacing / (2 * (np.sin(bin_spacing / 2)))  # Zar 1999, Equation 26.16
        complExp = [np.exp(per * 1j * d) for d in dirs]
        vector = np.dot(complExp, v)
        index = correction_factor * np.abs(vector)/np.sum(v)  # get the absolute of the vector, normalize to make it range between 0 and 1
        direction = cmath.phase(vector)/per
        #for orientation, the directions are mapped to the right half of a circle. Map instead to upper half
        if per == 2 and direction < 0:
            direction+=np.pi
        return index, direction

    def test_tuning(self, rep_dir_resps, dirs, per, iters=1000):
        """

        """
        (rep_n, dir_n) = rep_dir_resps.shape
        flattened = np.reshape(rep_dir_resps, (rep_n*dir_n))
        rand_idx = np.linspace(0, rep_n*dir_n-1, rep_n*dir_n, dtype=int)
        null_dist = np.zeros(iters)
        complExp = [np.exp(per * 1j * d) for d in dirs] / np.sqrt(len(dirs))
        q = np.abs(np.dot(complExp, rep_dir_resps.mean(axis=0)))
        for i in range(iters):
            np.random.shuffle(rand_idx)
            shuffled = flattened[rand_idx]
            shuffled = np.reshape(shuffled, (rep_n, dir_n))
            shuffled_mean = np.mean(shuffled, axis=0)
            null_dist[i] = np.abs(np.dot(complExp, shuffled_mean))

        return null_dist, np.mean(null_dist > q)