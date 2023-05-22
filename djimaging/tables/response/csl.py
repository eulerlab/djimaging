from abc import abstractmethod

from scipy.optimize import curve_fit
import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.response.response_quality import RepeatQITemplate
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.trace_utils import find_closest


def compute_csl_metrics(csl_trace, csl_time, triggertimes_rel, baseline_delay_s=0.5, response_delay_s=3.,
                        q_lb=5, q_ub=95):
    contrasts = [10, 20, 40, 60, 80, 100]

    # Find indexes
    idxs_baseline_start = np.zeros(len(contrasts), dtype=int)
    idxs_response_start = np.zeros(len(contrasts), dtype=int)
    idxs_response_end = np.zeros(len(contrasts), dtype=int)

    for contrast_idx, contrast in enumerate(contrasts):
        trigger_start = triggertimes_rel[contrast_idx]
        trigger_end = triggertimes_rel[contrast_idx + 1]

        idxs_baseline_start[contrast_idx] = find_closest(trigger_start + baseline_delay_s, csl_time, as_index=True)
        idxs_response_start[contrast_idx] = find_closest(trigger_start + response_delay_s, csl_time, as_index=True)
        idxs_response_end[contrast_idx] = find_closest(trigger_end, csl_time, as_index=True)

    assert np.all(idxs_baseline_start < idxs_response_start)
    assert np.all(idxs_response_start < idxs_response_end)

    # Get baselines
    baselines = []
    for i, (idx_a, idx_b) in enumerate(zip(idxs_baseline_start, idxs_response_start)):
        baselines.append(csl_trace[idx_a:idx_b])

    baselines = np.concatenate(baselines)
    baseline_mean = np.mean(baselines)
    baseline_std = np.std(baselines)

    # Normalize
    csl_norm = (csl_trace - baseline_mean) / baseline_std

    # Compute response metrics
    sds = np.full(len(contrasts), np.nan)
    lbs = np.full(len(contrasts), np.nan)
    ubs = np.full(len(contrasts), np.nan)

    for i, (idx_b, idx_c) in enumerate(zip(idxs_response_start, idxs_response_end)):
        sds[i] = np.std(csl_norm[idx_b:idx_c])
        lbs[i] = np.percentile(csl_norm[idx_b:idx_c], q_lb)
        ubs[i] = np.percentile(csl_norm[idx_b:idx_c], q_ub)

    return csl_norm, sds, lbs, ubs, idxs_baseline_start, idxs_response_start, idxs_response_end


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
        # Summary response to contrast stimulus
        -> self.averages_table
        ---
        csl_norm: longblob
        csl_sds: blob
        csl_lbs: blob
        csl_ubs: blob
        idxs_baseline_start: blob
        idxs_response_start: blob
        idxs_response_end: blob
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
        csl_trace = (self.averages_table & key).fetch1('average')
        csl_time = (self.averages_table & key).fetch1('average_times')
        triggertimes_rel = (self.averages_table & key).fetch1('triggertimes_rel')

        csl_norm, csl_sds, csl_lbs, csl_ubs, idxs_baseline_start, idxs_response_start, idxs_response_end = \
            compute_csl_metrics(csl_trace, csl_time, triggertimes_rel)

        key = key.copy()
        key['csl_norm'] = csl_norm
        key['csl_sds'] = csl_sds
        key['csl_lbs'] = csl_lbs
        key['csl_ubs'] = csl_ubs
        key['idxs_baseline_start'] = idxs_baseline_start
        key['idxs_response_start'] = idxs_response_start
        key['idxs_response_end'] = idxs_response_end
        self.insert1(key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        csl_norm, csl_sds, csl_lbs, csl_ubs, idxs_baseline_start, idxs_response_start, idxs_response_end = \
            (self & key).fetch1("csl_norm", "csl_sds", "csl_lbs", "csl_ubs",
                                "idxs_baseline_start", "idxs_response_start", "idxs_response_end")

        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex='all')

        ax = axs[0]
        ax.set(ylabel='diffs')
        ax.plot(0.5 * (idxs_response_start + idxs_response_end), csl_sds, '.-')

        ax = axs[1]
        ax.set(ylabel='trace')
        ax.plot(csl_norm, c='k')

        for lb, ub, idx_a, idx_b, idx_c in zip(
                csl_lbs, csl_ubs, idxs_baseline_start, idxs_response_start, idxs_response_end):
            ax.plot([idx_a, idx_b], [0, 0], c='gray')
            ax.plot([idx_b, idx_c], [lb, lb], c='blue')
            ax.plot([idx_b, idx_c], [ub, ub], c='red')
            ax.plot([0.5 * (idx_b + idx_c), 0.5 * (idx_b + idx_c)], [lb, ub], c='orange', ls='--')

        plt.show()


class CslSlopeTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Fit slope to contrast responses
        -> self.csl_features_table
        ---
        csl_slope: float
        csl_intercept: float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.csl_features_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def csl_features_table(self):
        pass

    def make(self, key):
        csl_sds = (self.csl_features_table & key).fetch1('csl_sds')
        csl_slope, csl_intercept = fit_linear(csl_sds)
        key = key.copy()
        key['csl_slope'] = csl_slope
        key['csl_intercept'] = csl_intercept
        self.insert1(key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        csl_slope, csl_intercept = (self & key).fetch1('csl_slope', 'csl_intercept')

        csl_norm, csl_sds, csl_lbs, csl_ubs, idxs_baseline_start, idxs_response_start, idxs_response_end = \
            (self.csl_features_table & key).fetch1("csl_norm", "csl_sds", "csl_lbs", "csl_ubs",
                                                   "idxs_baseline_start", "idxs_response_start", "idxs_response_end")

        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex='all')

        fig.suptitle(key, y=1, va='bottom')

        ax = axs[0]
        ax.set(ylabel='diffs', title=f'slope={csl_slope:.1f}, y0={csl_intercept:.1f}')
        ax.plot(0.5 * (idxs_response_start + idxs_response_end), csl_sds, '.-')
        ax.plot(0.5 * (idxs_response_start + idxs_response_end),
                linear(np.linspace(0, 1, csl_sds.size), a=csl_slope, b=csl_intercept), '.-')

        ax = axs[1]
        ax.set(ylabel='trace')
        ax.plot(csl_norm, c='k')

        for lb, ub, idx_a, idx_b, idx_c in zip(
                csl_lbs, csl_ubs, idxs_baseline_start, idxs_response_start, idxs_response_end):
            ax.plot([idx_a, idx_b], [0, 0], c='gray')
            ax.plot([idx_b, idx_c], [lb, lb], c='blue')
            ax.plot([idx_b, idx_c], [ub, ub], c='red')
            ax.plot([0.5 * (idx_b + idx_c), 0.5 * (idx_b + idx_c)], [lb, ub], c='orange', ls='--')

        plt.show()


class CslQITemplate(RepeatQITemplate):
    _stim_family = "csl"
    _stim_name = "csl"

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass
