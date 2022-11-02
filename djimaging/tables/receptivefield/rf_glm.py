import warnings
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.receptivefield.rf_glm_utils import compute_glm_receptive_field, plot_rf_summary, quality_test
from djimaging.utils.dj_utils import PlaceholderTable, get_primary_key

try:
    import rfest
except ImportError:
    warnings.warn('Did not find RFEst package. Install it to estimates RFs.')
    rfest = None


class RFGLMParamsTemplate(dj.Lookup):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        rf_glm_params_id: int # unique param set id
        ---
        dur_filter_s: float # minimum duration of filter in seconds
        
        df_ts : blob
        df_ws : blob
        betas : blob

        norm_stim : tinyint unsigned  # Normalize (znorm) stimulus?
        norm_trace : tinyint unsigned  # Normalize (znorm) trace?
        filter_trace: tinyint unsigned  # Low pass filter trace?
        cutoff = 10: float  # Cutoff frequency low pass filter

        kfold : tinyint unsigned        
        
        min_iters = 50: int unsigned 
        max_iters = 1000: int unsigned
        step_size = 0.1: float
        tolerance = 5: int unsigned
        alpha = 1: float
        metric = "mse": enum("mse", "corrcoef")
        fit_verbose = 0 : int unsigned
        """
        return definition

    def add_default(self, skip_duplicates=False, **update_kw):
        """Add default preprocess parameter to table"""
        key = {
            'rf_glm_params_id': 1,
            'dur_filter_s': 1.0,
            'df_ts': [8],
            'df_ws': [9],
            'betas': [0.001, 0.01, 0.1],
            'norm_stim': 1,
            'norm_trace': 1,
            'kfold': 1,
        }
        key.update(**update_kw)
        self.insert1(key, skip_duplicates=skip_duplicates)


class RFGLMTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.preprocesstraces_table
        -> self.params_table
        ---
        rf: longblob  # spatio-temporal receptive field
        dt: float
        model_dict: longblob
        quality_dict: longblob
        '''
        return definition

    params_table = PlaceholderTable
    preprocesstraces_table = PlaceholderTable
    stimulus_table = PlaceholderTable
    presentation_table = PlaceholderTable
    traces_table = PlaceholderTable

    @property
    def key_source(self):
        try:
            return self.params_table() * self.preprocesstraces_table() & \
                   (self.stimulus_table() & "stim_family = 'noise'")
        except TypeError:
            pass

    def make(self, key):
        stim = (self.stimulus_table() & key).fetch1("stim_trace")
        stimtime = (self.presentation_table() & key).fetch1('triggertimes')
        assert stim is not None, "stim_trace in stimulus table must be set."

        if stim.shape[0] != stimtime.size:
            warnings.warn(f'Stimulus length ({stim.shape[0]}) does not match stimtime ({stimtime.size}).')
            stim = stim[:stimtime.size]

        trace, tracetime = (self.preprocesstraces_table() & key).fetch1('preprocess_trace', 'preprocess_trace_times')

        params = (self.params_table() & key).fetch1()

        rf, quality_dict, model_dict = compute_glm_receptive_field(
            trace=trace, tracetime=tracetime, stim=stim, triggertime=stimtime,
            dur_tfilter=params['dur_filter_s'], df_ts=params['df_ts'], df_ws=params['df_ws'],
            betas=params['betas'], kfold=params['kfold'], alpha=params['alpha'], metric=params['metric'],
            tolerance=params['tolerance'], verbose=params['fit_verbose'],
            norm_stim=params['norm_trace'], norm_trace=params['norm_trace'],
            filter_trace=params['filter_trace'], cutoff=params['cutoff'],
            min_iters=params['min_iters'], max_iters=params['max_iters'], step_size=params['step_size'],
            p_keep=1., gradient=False, output_nonlinearity='none', fupsample=0, init_method=None,
            n_perm=20, min_cc=0.2, seed=42, logger=None, fit_R=False, fit_intercept=True)

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
        key = {
            'glm_quality_params_id': 1,
            'min_corrcoef': 0.1,
            'max_mse': 1.0,
            'perm_alpha': 0.001,
        }
        key.update(**update_kw)
        self.insert1(key, skip_duplicates=skip_duplicates)


class RFGLMQualityTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
            # Compute basic receptive fields
            -> self.glm_table
            -> self.params_table
            ---
            rf_glm_qidx : float  
            '''
        return definition

    glm_table = PlaceholderTable
    params_table = PlaceholderTable

    def make(self, key):
        quality_dict, model_dict = (self.glm_table() & key).fetch1('quality_dict', 'model_dict')
        min_corrcoef, max_mse, perm_alpha = (self.params_table() & key).fetch1(
            'min_corrcoef', 'max_mse', 'perm_alpha')

        p_value = quality_test(
            score_trueX=quality_dict['score_test'],
            score_permX=quality_dict['score_permX'],
            metric=model_dict['metric'])

        rf_glm_qidx_perm = p_value <= perm_alpha

        rf_glm_qidx_corrcoeff = \
            (quality_dict['corrcoef_train'] >= min_corrcoef) & \
            (quality_dict.get('corrcoef_dev', 1.) >= min_corrcoef) & \
            (quality_dict['corrcoef_test'] >= min_corrcoef)

        rf_glm_qidx_mse = \
            (quality_dict['mse_train'] <= max_mse) & \
            (quality_dict.get('mse_dev', 1.) <= max_mse) & \
            (quality_dict['mse_test'] <= max_mse)

        rf_glm_qidx = \
            np.around((float(rf_glm_qidx_perm) + float(rf_glm_qidx_corrcoeff) + float(rf_glm_qidx_mse)) / 3., 2)

        q_key = key.copy()
        q_key['rf_glm_qidx'] = float(rf_glm_qidx)

        self.insert1(q_key)

    def plot(self, glm_quality_params_id=1):
        rf_glm_qidx = (self & f"glm_quality_params_id={glm_quality_params_id}").fetch("rf_glm_qidx")

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.hist(rf_glm_qidx)
        ax.set(title= f"glm_quality_params_id={glm_quality_params_id}", ylabel="rf_glm_qidx")
        plt.show()
