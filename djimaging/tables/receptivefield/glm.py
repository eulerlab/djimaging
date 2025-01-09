"""
Receptive field estimation using GLMs.

Example usage:

from djimaging.tables import receptivefield, rf_glm

@schema
class DNoiseTraceParams(receptivefield.DNoiseTraceParamsTemplate):
    pass


@schema
class DNoiseTrace(receptivefield.DNoiseTraceTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    traces_table = PreprocessTraces
    params_table = DNoiseTraceParams


@schema
class RfGlmParams(rf_glm.RfGlmParamsTemplate):
    pass


@schema
class RfGlm(rf_glm.RfGlmTemplate):
    noise_traces_table = DNoiseTrace
    params_table = RfGlmParams
    preprocesstraces_table = PreprocessTraces
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces


@schema
class RfGlmSingleModel(rf_glm.RfGlmSingleModelTemplate):
    noise_traces_table = DNoiseTrace
    glm_table = RfGlm


@schema
class RfGlmQualityParams(rf_glm.RfGlmQualityParamsTemplate):
    pass


@schema
class RfGlmQuality(rf_glm.RfGlmQualityTemplate):
    glm_single_model_table = RfGlmSingleModel
    glm_table = RfGlm
    params_table = RfGlmQualityParams


@schema
class SplitRfGlmParams(receptivefield.SplitRFParamsTemplate):
    pass


@schema
class SplitRfGlm(receptivefield.SplitRFTemplate):
    rf_table = RfGlm
    split_rf_params_table = SplitRfGlmParams


@schema
class FitGauss2DRfGlm(receptivefield.FitGauss2DRFTemplate):
    split_rf_table = SplitRfGlm
    stimulus_table = Stimulus


@schema
class FitDoG2DRfGlm(receptivefield.FitDoG2DRFTemplate):
    split_rf_table = SplitRfGlm
    stimulus_table = Stimulus
"""

import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.receptive_fields.plot_rf_utils import plot_rf_frames, plot_rf_video
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.receptive_fields.glm_utils import ReceptiveFieldGLM, plot_rf_summary, quality_test


class RfGlmParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        rf_glm_params_id: tinyint unsigned # unique param set id
        ---
        filter_dur_s_past : float # filter duration in seconds into the past
        filter_dur_s_future : float # filter duration in seconds into the future
        df_ts : blob
        df_ws : blob
        betas : blob
        kfold : tinyint unsigned        
        metric = "mse": enum("mse", "corrcoef")
        output_nonlinearity = 'none' : varchar(63)
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


class RfGlmTemplate(dj.Computed):
    database = ""
    _def_sta = True  # If True, the definition includes rf_time

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.noise_traces_table
        -> self.params_table
        ---
        rf: longblob  # spatio-temporal receptive field
        dt: float
        '''

        if self._def_sta:
            definition += '''
            rf_time: longblob
            shift: int
            '''

        definition += '''
        model_dict: longblob
        quality_dict: longblob
        '''

        return definition

    @property
    def key_source(self):
        try:
            return self.noise_traces_table.proj() * self.params_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def noise_traces_table(self):
        pass

    def make(self, key, suppress_outputs=False, clear_outputs=True):
        params = (self.params_table() & key).fetch1()
        noise_dt, noise_t0, trace, stim_idxs = (self.noise_traces_table() & key).fetch1(
            'noise_dt', 'noise_t0', 'trace', 'stim_idxs')
        assert trace.size == stim_idxs.size, "Trace and stim_idxs must have the same size."
        stim = (self.noise_traces_table.stimulus_table() & key).fetch1("stim_trace")
        stim = stim[stim_idxs].astype(trace.dtype)

        other_params_dict = params.pop('other_params_dict')
        if suppress_outputs:
            other_params_dict['verbose'] = 0

        model = ReceptiveFieldGLM(
            dt=noise_dt, trace=trace, stim=stim,
            filter_dur_s_past=params['filter_dur_s_past'],
            filter_dur_s_future=params['filter_dur_s_future'],
            df_ts=params['df_ts'], df_ws=params['df_ws'],
            betas=params['betas'], kfold=params['kfold'], metric=params['metric'],
            output_nonlinearity=params['output_nonlinearity'], **other_params_dict)

        rf, quality_dict, model_dict = model.compute_glm_receptive_field()

        if clear_outputs or suppress_outputs:
            from IPython.display import clear_output
            clear_output(wait=True)

        if not suppress_outputs:
            plot_rf_summary(rf=rf, quality_dict=quality_dict, model_dict=model_dict,
                            title=f"{key['date']} {key['exp_num']} {key['field']} {key['roi_id']}")
            plt.show()

        rf_key = deepcopy(key)
        rf_key['rf'] = rf.astype(np.float32)
        rf_key['dt'] = model_dict.pop('dt')

        if self._def_sta:
            rf_key['rf_time'] = model_dict.pop('rf_time')
            rf_key['shift'] = model_dict['shift']['stimulus']  # There may be other shifts

        rf_key['model_dict'] = model_dict
        rf_key['quality_dict'] = quality_dict

        self.insert1(rf_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        rf, quality_dict, model_dict = (self & key).fetch1('rf', 'quality_dict', 'model_dict')
        plot_rf_summary(rf=rf, quality_dict=quality_dict, model_dict=model_dict,
                        title=f"{key['date']} {key['exp_num']} {key['field']} {key['roi_id']}")
        plt.show()

    def plot1_frames(self, key=None, downsample=1):
        key = get_primary_key(table=self, key=key)
        rf, model_dict = (self & key).fetch1('rf', 'model_dict')
        plot_rf_frames(rf, model_dict['rf_time'], downsample=downsample)

    def plot1_video(self, key=None, fps=10):
        key = get_primary_key(table=self, key=key)
        rf, model_dict = (self & key).fetch1('rf', 'model_dict')
        return plot_rf_video(rf, model_dict['rf_time'], fps=fps)


class RfGlmQualityDictTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
            -> self.glm_table
            ---
            corrcoef_train = NULL : float
            corrcoef_test = NULL : float
            corrcoef_dev = NULL : float
            mse_train = NULL : float
            mse_dev = NULL : float
            mse_test = NULL : float
            '''
        return definition

    @property
    def key_source(self):
        try:
            return self.glm_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def glm_table(self):
        pass

    def make(self, key):
        quality_dict = (self.glm_table & key).fetch1('quality_dict')
        self.insert1(dict(
            **key,
            corrcoef_train=quality_dict.get('corrcoef_train', np.nan),
            corrcoef_dev=quality_dict.get('corrcoef_dev', np.nan),
            corrcoef_test=quality_dict.get('corrcoef_test', np.nan),
            mse_train=quality_dict.get('mse_train', np.nan),
            mse_dev=quality_dict.get('mse_dev', np.nan),
            mse_test=quality_dict.get('mse_test', np.nan),
        ))

    def plot(self, *restrictions):
        df_q = (self & restrictions).fetch(format='frame')


class RfGlmQualityParamsTemplate(dj.Lookup):
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


class RfGlmSingleModelTemplate(dj.Computed):
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
    def key_source(self):
        try:
            return self.glm_table.noise_traces_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def glm_table(self):
        pass

    def make(self, key):
        roi_key = key.copy()
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


class RfGlmQualityTemplate(dj.Computed):
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
    def key_source(self):
        try:
            return self.glm_single_model_table().proj() * self.params_table().proj()
        except (AttributeError, TypeError):
            pass

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

        rf_glm_qidx_sum = 0.
        rf_glm_qidx_count = 0

        if perm_alpha > 0:
            p_value = quality_test(
                score_trueX=quality_dict['score_test'],
                score_permX=quality_dict['score_permX'],
                metric=model_dict['metric'])

            rf_glm_qidx_perm = p_value <= perm_alpha
            rf_glm_qidx_sum += float(rf_glm_qidx_perm)
            rf_glm_qidx_count += 1

        if min_corrcoef > 0:
            rf_glm_qidx_corrcoeff = \
                (quality_dict['corrcoef_train'] >= min_corrcoef) & \
                (quality_dict.get('corrcoef_dev', np.inf) >= min_corrcoef) & \
                (quality_dict['corrcoef_test'] >= min_corrcoef)
            rf_glm_qidx_sum += float(rf_glm_qidx_corrcoeff)
            rf_glm_qidx_count += 1

        if max_mse > 0:
            rf_glm_qidx_mse = \
                (quality_dict['mse_train'] <= max_mse) & \
                (quality_dict.get('mse_dev', np.NINF) <= max_mse) & \
                (quality_dict['mse_test'] <= max_mse)
            rf_glm_qidx_sum += float(rf_glm_qidx_mse)
            rf_glm_qidx_count += 1

        rf_glm_qidx = rf_glm_qidx_sum / rf_glm_qidx_count

        q_key = key.copy()
        q_key['rf_glm_qidx'] = rf_glm_qidx

        self.insert1(q_key)

    def plot(self, glm_quality_params_id=1):
        rf_glm_qidx = (self & f"glm_quality_params_id={glm_quality_params_id}").fetch("rf_glm_qidx")

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.hist(rf_glm_qidx)
        ax.set(title=f"glm_quality_params_id={glm_quality_params_id}", ylabel="rf_glm_qidx")
        ax.set_xlim(-0.1, 1.1)
        plt.show()
