from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt


class RoiQualityParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = f"""
        quality_params_id : tinyint unsigned
        ---
        combination: enum("or", "and")
        min_qidx_gchirp : float
        min_qidx_lchirp : float
        """
        return definition

    def add(self, params_id=1, min_qidx_gchirp=0.35, min_qidx_lchirp=0.35, skip_duplicates=False):
        key = dict(
            quality_params_id=params_id, min_qidx_gchirp=min_qidx_gchirp, min_qidx_lchirp=min_qidx_lchirp)
        self.insert1(key, skip_duplicates=skip_duplicates)


class RoiQualityTemplate(dj.Computed):
    """Quality template that combines different quality measurements.
    In many cases this class needs to be modified to also take other stimuli into account.
    TODO: implement more universal strategy to add new stimuli.
    """
    database = ""

    @property
    def definition(self):
        definition = f'''
        -> self.params_table
        -> self.roi_table
        ---
        q_tot : tinyint unsigned  # 1: Use this ROI, 0: Don't use it.
        q_gchirp : tinyint  # 1: Good response, 0: bad response, -1: No or bad recording
        q_lchirp : tinyint  # 1: Good response, 0: bad response, -1: No or bad recording
        '''
        return definition

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def chirp_qi_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.roi_table & "artifact_flag=0").proj() * self.params_table.proj()
        except (AttributeError, TypeError):
            pass

    @staticmethod
    def get_quality(qidx, min_qidx):
        qidx = np.atleast_1d(qidx)
        if len(qidx) > 1:
            raise ValueError('Found multiple qidx where at most one was expected')

        if len(qidx) == 0:
            return -1
        elif qidx[0] >= min_qidx:
            return 1
        else:
            return 0

    def make(self, key):
        min_qidx_gchirp, min_qidx_lchirp, combination = (self.params_table & key).fetch1(
            "min_qidx_gchirp", "min_qidx_lchirp", "combination")

        q_gchirp = self.get_quality(
            (self.chirp_qi_table() & "stim_name='gChirp'" & key).fetch("qidx"), min_qidx=min_qidx_gchirp)
        q_lchirp = self.get_quality(
            (self.chirp_qi_table() & "stim_name='lChirp'" & key).fetch("qidx"), min_qidx=min_qidx_lchirp)

        if combination == 'or':
            q_tot = (q_gchirp == 1) or (q_lchirp == 1)
        elif combination == 'and':
            q_tot = (q_gchirp == 1) and (q_lchirp == 1)
        else:
            raise NotImplementedError(combination)

        key = key.copy()
        key['q_tot'] = q_tot
        key['q_gchirp'] = q_gchirp
        key['q_lchirp'] = q_lchirp
        self.insert1(key)

    def plot(self):
        def plot_qidx(ax_, qidxs_, title):
            qidxs_ = np.asarray(qidxs_).astype(int)
            bins = np.arange(np.min(qidxs_) - 0.25, np.max(qidxs_) + 0.5, 0.5)
            ax_.hist(qidxs_, bins=bins)
            ax_.set_xticks(np.unique(qidxs_))
            ax_.set_title(title)

        fig, axs = plt.subplots(1, 4, figsize=(12, 2))
        plot_qidx(ax_=axs[0], qidxs_=self.fetch('q_tot'), title='q_tot')
        plot_qidx(ax_=axs[1], qidxs_=self.fetch('q_gchirp'), title='q_gchirp')
        plot_qidx(ax_=axs[2], qidxs_=self.fetch('q_lchirp'), title='q_lchirp')
        plt.show()
