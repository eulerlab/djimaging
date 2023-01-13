import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.rf_glm.rf_glm_utils import ReceptiveFieldGLM, plot_rf_summary, quality_test
from djimaging.utils.dj_utils import get_primary_key


class RFGLMParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        rf_glm_params_id: int # unique param set id
        ---
        filter_dur_s_past : float # filter duration in seconds into the past
        filter_dur_s_future : float # filter duration in seconds into the future
        df_ts : blob
        df_ws : blob
        betas : blob
        kfold : tinyint unsigned        
        metric = "mse": enum("mse", "corrcoef")
        output_nonlinearity = 'none' : varchar(255)
        other_params_dict : longblob
        """
        return definition

    def add_default(self, other_params_dict=None, skip_duplicates=False, **params):
        """Add default preprocess parameter to table"""

        assert "other_params_dict" not in params.keys()
        other_params_dict = dict() if other_params_dict is None else other_params_dict

        other_params_dict_default = dict(
            min_iters=50, max_iters=1000, step_size=0.1, tolerance=5,
            alphas=(1.,), verbose=0, n_perm=20, min_cc=0.2, seed=42,
            fit_R=False, fit_intercept=True, init_method=None, atol=1e-3,
            distr='gaussian'
        )

        other_params_dict_default.update(**other_params_dict)

        key = dict(
            rf_glm_params_id=1,
            filter_dur_s_past=1.,
            filter_dur_s_future=0.,
            df_ts=(8,), df_ws=(9,),
            betas=(0.001, 0.01, 0.1), kfold=1,
            other_params_dict=other_params_dict_default,
        )

        key.update(**params)
        self.insert1(key, skip_duplicates=skip_duplicates)


class RFGLMTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.noise_traces_table
        -> self.params_table
        ---
        rf: longblob  # spatio-temporal receptive field
        dt: float
        model_dict: longblob
        quality_dict: longblob
        '''
        return definition

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def noise_traces_table(self):
        pass

    def make(self, key):
        params = (self.params_table() & key).fetch1()
        other_params_dict = params.pop('other_params_dict')
        dt, time, trace, stim = (self.noise_traces_table() & key).fetch1('dt', 'time', 'trace', 'stim')

        model = ReceptiveFieldGLM(
            dt=dt, trace=trace, stim=stim,
            filter_dur_s_past=params['filter_dur_s_past'], filter_dur_s_future=params['filter_dur_s_future'],
            df_ts=params['df_ts'], df_ws=params['df_ws'],
            betas=params['betas'], kfold=params['kfold'], metric=params['metric'],
            output_nonlinearity=params['output_nonlinearity'], **other_params_dict)

        rf, quality_dict, model_dict = model.compute_glm_receptive_field()

        from IPython.display import clear_output
        clear_output(wait=True)

        plot_rf_summary(rf=rf, quality_dict=quality_dict, model_dict=model_dict,
                        title=f"{key['date']} {key['exp_num']} {key['field']} {key['roi_id']}")
        plt.show()

        rf_key = deepcopy(key)
        rf_key['rf'] = rf
        rf_key['dt'] = model_dict.pop('dt')
        rf_key['model_dict'] = model_dict
        rf_key['quality_dict'] = quality_dict

        self.insert1(rf_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        rf, quality_dict, model_dict = (self & key).fetch1('rf', 'quality_dict', 'model_dict')
        plot_rf_summary(rf=rf, quality_dict=quality_dict, model_dict=model_dict,
                        title=f"{key['date']} {key['exp_num']} {key['field']} {key['roi_id']}")
        plt.show()


class RFGLMQualityParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        glm_quality_params_id: smallint # unique param set id
        ---
        min_corrcoef : float
        max_mse : float
        perm_alpha : float
        """
        return definition

    def add_default(self, skip_duplicates=False, **update_kw):
        """Add default preprocess parameter to table"""
        key = dict(
            glm_quality_params_id=1,
            min_corrcoef=0.1,
            max_mse=1.0,
            perm_alpha=0.001,
        )
        key.update(**update_kw)
        self.insert1(key, skip_duplicates=skip_duplicates)


class RFGLMSingleModelTemplate(dj.Computed):
    database = ""
    _default_metric = 'mse'

    @property
    def definition(self):
        definition = '''
            # Pick best model for all parameterizations
            -> self.glm_table
            ---
            score_test : float
            '''
        return definition

    @property
    @abstractmethod
    def glm_table(self):
        pass

    def make(self, key):
        roi_key = key.copy()
        roi_key.pop('rf_glm_params_id')
        quality_dicts, model_dicts, param_ids = (self.glm_table & roi_key).fetch(
            'quality_dict', 'model_dict', 'rf_glm_params_id')

        metrics = [model_dict['metric'] for model_dict in model_dicts]
        if np.unique(metrics).size == 1:
            metric = metrics[0]
        else:
            warnings.warn(f'Different metric were used. Fallback to {self._default_metric}')
            metric = self._default_metric

        scores_test = [quality_dict[f'{metric}_test'] for quality_dict in quality_dicts]

        # Do some sanity check
        if np.unique(metrics).size == 1:
            for i, score_test in enumerate(scores_test):
                assert np.isclose(quality_dicts[i]['score_test'], score_test)

        # Get best model parameterization
        if metric in ['mse', 'gcv']:
            best_i = np.argmin(scores_test)
        else:
            best_i = np.argmax(scores_test)

        score_test = scores_test[best_i]
        param_id = param_ids[best_i]

        if len(self & roi_key) > 1:
            raise ValueError(f'Already more than one entry present for key={roi_key}. This should not happen.')
        elif len(self & roi_key) == 1:
            (self & roi_key).delete_quick()  # Delete for update, e.g. if new param_ids were added

        key = key.copy()
        key['rf_glm_params_id'] = param_id
        key['score_test'] = score_test
        self.insert1(key)

    def plot(self):
        rf_glm_params_ids = self.fetch('rf_glm_params_id')
        plt.figure()
        plt.hist(rf_glm_params_ids,
                 bins=np.arange(np.min(rf_glm_params_ids) - 0.25, np.max(rf_glm_params_ids) + 0.5, 0.5))
        plt.title('rf_glm_params_id')
        plt.show()

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        rf, quality_dict, model_dict = (self.glm_table & key).fetch1('rf', 'quality_dict', 'model_dict')
        plot_rf_summary(rf=rf, quality_dict=quality_dict, model_dict=model_dict,
                        title=f"{key['date']} {key['exp_num']} {key['field']} {key['roi_id']}")
        plt.show()


class RFGLMQualityTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
            # Compute basic receptive fields
            -> self.glm_single_model_table
            -> self.params_table
            ---
            rf_glm_qidx : float  
            '''
        return definition

    @property
    @abstractmethod
    def glm_single_model_table(self):
        pass

    @property
    @abstractmethod
    def glm_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    def make(self, key):
        quality_dict, model_dict = (self.glm_table() & key).fetch1('quality_dict', 'model_dict')
        min_corrcoef, max_mse, perm_alpha = (self.params_table() & key).fetch1(
            'min_corrcoef', 'max_mse', 'perm_alpha')

        if perm_alpha > 0:
            p_value = quality_test(
                score_trueX=quality_dict['score_test'],
                score_permX=quality_dict['score_permX'],
                metric=model_dict['metric'])

            rf_glm_qidx_perm = p_value <= perm_alpha
        else:
            rf_glm_qidx_perm = None

        rf_glm_qidx_corrcoeff = \
            (quality_dict['corrcoef_train'] >= min_corrcoef) & \
            (quality_dict.get('corrcoef_dev', 1.) >= min_corrcoef) & \
            (quality_dict['corrcoef_test'] >= min_corrcoef)

        rf_glm_qidx_mse = \
            (quality_dict['mse_train'] <= max_mse) & \
            (quality_dict.get('mse_dev', 1.) <= max_mse) & \
            (quality_dict['mse_test'] <= max_mse)

        if perm_alpha > 0:
            rf_glm_qidx = \
                np.around((float(rf_glm_qidx_perm) + float(rf_glm_qidx_corrcoeff) + float(rf_glm_qidx_mse)) / 3., 2)
        else:
            rf_glm_qidx = np.around((float(rf_glm_qidx_corrcoeff) + float(rf_glm_qidx_mse)) / 2., 2)

        q_key = key.copy()
        q_key['rf_glm_qidx'] = rf_glm_qidx

        self.insert1(q_key)

    def plot(self, glm_quality_params_id=1):
        rf_glm_qidx = (self & f"glm_quality_params_id={glm_quality_params_id}").fetch("rf_glm_qidx")

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.hist(rf_glm_qidx)
        ax.set(title=f"glm_quality_params_id={glm_quality_params_id}", ylabel="rf_glm_qidx")
        plt.show()
