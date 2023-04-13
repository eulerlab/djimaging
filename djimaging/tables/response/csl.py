from abc import abstractmethod

from scipy.optimize import curve_fit
import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
from djimaging.utils.dj_utils import get_primary_key


def compute_csl_metrics(csl_norm):
    # TODO: Improve feature computation.

    # calculate min_max_difference for every contrast
    # continue analysis with csl_norm (normalized csl traces with corrected baseline)
    contrasts = [10, 20, 40, 60, 80, 100]

    idx_as = np.zeros(len(contrasts), dtype=int)
    idx_bs = np.zeros(len(contrasts), dtype=int)
    responses = np.full(len(contrasts), np.nan)
    lbs = np.full(len(contrasts), np.nan)
    ubs = np.full(len(contrasts), np.nan)

    for contrast_idx, contrast in enumerate(contrasts):
        idx_a = contrast_idx * 90 + 30
        idx_b = idx_a + 60

        responses[contrast_idx] = np.std(csl_norm[idx_a:idx_b])

        lbs[contrast_idx] = np.percentile(csl_norm[idx_a:idx_b], 5)
        ubs[contrast_idx] = np.percentile(csl_norm[idx_a:idx_b], 95)

        idx_as[contrast_idx] = idx_a
        idx_bs[contrast_idx] = idx_b

    return responses, lbs, ubs, idx_as, idx_bs


def linear(x, a, b):
    return a * x + b


def fit_linear(csl_responses):
    slope, intercept = curve_fit(
        f=linear, xdata=np.linspace(0, 1, csl_responses.size, endpoint=True), ydata=csl_responses)[0]
    return slope, intercept


class CslFeaturesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        Summary response to constrast stimulus
        -> self.averages_table
        ---
        csl_responses: longblob
        csl_lbs: longblob
        csl_ubs: longblob
        idx_as: longblob
        idx_bs: longblob
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.averages_table().proj() & \
                (self.stimulus_table() & "stim_name = 'csl' or stim_family = 'csl'")
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def averages_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def make(self, key):
        csl_norm_data = (self.averages_table & key).fetch1('average_norm')
        csl_responses, csl_lbs, csl_ubs, idx_as, idx_bs = compute_csl_metrics(csl_norm_data)

        key = key.copy()
        key['csl_responses'] = csl_responses
        key['csl_lbs'] = csl_lbs
        key['csl_ubs'] = csl_ubs
        key['idx_as'] = idx_as
        key['idx_bs'] = idx_bs
        self.insert1(key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        csl_norm_data = (self.normalized_csl_table & key).fetch1('normalized_csl')

        csl_responses, csl_lbs, csl_ubs, idx_as, idx_bs = (self & key).fetch1(
            "csl_responses", "csl_lbs", "csl_ubs", "idx_as", "idx_bs")

        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex='all')

        ax = axs[0]
        ax.set(ylabel='diffs')
        ax.plot(0.5 * (idx_as + idx_bs), csl_responses, '.-')

        ax = axs[1]
        ax.set(ylabel='trace')
        ax.plot(csl_norm_data, c='k')

        for lb, ub, idx_a, idx_b in zip(csl_lbs, csl_ubs, idx_as, idx_bs):
            ax.plot([idx_a, idx_b], [lb, lb], c='blue')
            ax.plot([idx_a, idx_b], [ub, ub], c='red')
            ax.plot([0.5 * (idx_a + idx_b), 0.5 * (idx_a + idx_b)], [lb, ub], c='orange', ls='--')

        plt.show()


class CslSlopeTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        Fit slope to contrast responses
        -> self.csl_features_table
        ---
        csl_slope: float
        csl_intercept: float
        """
        return definition

    @property
    @abstractmethod
    def csl_features_table(self):
        pass

    def make(self, key):
        csl_responses = (self.csl_features_table & key).fetch1('csl_responses')
        csl_slope, csl_intercept = fit_linear(csl_responses)
        key = key.copy()
        key['csl_slope'] = csl_slope
        key['csl_intercept'] = csl_intercept
        self.insert1(key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        csl_slope, csl_intercept = (self & key).fetch1('csl_slope', 'csl_intercept')
        csl_norm_data = (self.csl_features_table.normalized_csl_table & key).fetch1('normalized_csl')

        csl_responses, csl_lbs, csl_ubs, idx_as, idx_bs = (self.csl_features_table & key).fetch1(
            "csl_responses", "csl_lbs", "csl_ubs", "idx_as", "idx_bs")

        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex='all')

        fig.suptitle(key, y=1, va='bottom')

        ax = axs[0]
        ax.set(ylabel='diffs')
        ax.plot(0.5 * (idx_as + idx_bs), csl_responses, '.-')
        ax.plot(0.5 * (idx_as + idx_bs),
                linear(np.linspace(0, 1, csl_responses.size), a=csl_slope, b=csl_intercept), '.-')

        ax = axs[1]
        ax.set(ylabel='trace')
        ax.plot(csl_norm_data, c='k')

        for lb, ub, idx_a, idx_b in zip(csl_lbs, csl_ubs, idx_as, idx_bs):
            ax.plot([idx_a, idx_b], [lb, lb], c='blue')
            ax.plot([idx_a, idx_b], [ub, ub], c='red')
            ax.plot([0.5 * (idx_a + idx_b), 0.5 * (idx_a + idx_b)], [lb, ub], c='orange', ls='--')

        plt.show()
