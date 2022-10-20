from copy import deepcopy

import datajoint as dj

from djimaging.tables.special.rf_glm_utils import compute_glm_receptive_field
from djimaging.utils.dj_utils import PlaceholderTable

try:
    import rfest
except ImportError:
    import warnings

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
            'betas': [0.001, 0.002, 0.006, 0.01, 0.02, 0.06, 0.1],
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

        trace = (self.preprocesstraces_table() & key).fetch1('preprocess_trace')
        tracetime = (self.traces_table() & key).fetch1('trace_times')

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

        rf_key = deepcopy(key)
        rf_key['rf'] = rf
        rf_key['model_dict'] = model_dict
        rf_key['quality_dict'] = quality_dict

        self.insert1(rf_key)


