"""
Receptive field estimation using GLMs.

The tables follow the logic of the fast STA tables (see `fast_sta.py`): The stimulus design matrix is stored in the
parameter table and built once per stimulus. All traces of a presentation are then aligned to the stimulus and fit
without any intermediate table. The tables subclass the fast STA templates and only replace the fitting step.

For speed, the projections of the design matrix onto the spline bases are computed once per `populate`, and the traces
of a presentation are fit in parallel processes. Use `populate(n_jobs=...)` to set the number of CPU cores.

Example usage:

from djimaging.tables import receptivefield

@schema
class RfGlmParams(receptivefield.RfGlmParamsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation


@schema
class RfGlm(receptivefield.RfGlmTemplate):
    params_table = RfGlmParams
    presentation_table = Presentation
    traces_table = PreprocessTraces


@schema
class RfGlmQualityParams(receptivefield.RfGlmQualityParamsTemplate):
    pass


@schema
class RfGlmQuality(receptivefield.RfGlmQualityTemplate):
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

import os
import shutil
import tempfile
import weakref
from abc import abstractmethod

import datajoint as dj
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from djimaging.tables.receptivefield.fast_sta import FastStaParamsTemplate, FastStaTemplate
from djimaging.utils.dj_storage import load_array
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.receptive_fields.glm_utils import ReceptiveFieldGLM, fit_receptive_field_glm, \
    plot_rf_summary, quality_test


def _rf_to_channels_first(rf: np.ndarray, model_dict: dict) -> np.ndarray:
    """Move the color axis of a stored RF (time, ..., color) to the front, as expected by `plot_rf_summary`."""
    if model_dict.get('n_colors', 1) > 1:
        return np.moveaxis(rf, -1, 0)
    return rf


class _MemmappedArrays(dict):
    """Nested dict of read-only memory-mapped arrays, whose files are deleted when the dict is garbage collected.

    Worker processes open memory-mapped arrays by file name instead of receiving a copy.
    """

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str = 'djimaging_') -> '_MemmappedArrays':
        folder = tempfile.mkdtemp(prefix=prefix)
        obj = cls()
        weakref.finalize(obj, shutil.rmtree, folder, True)
        for i, (k1, arrays_k1) in enumerate(arrays.items()):
            obj[k1] = dict()
            for j, (k2, array) in enumerate(arrays_k1.items()):
                file = os.path.join(folder, f'{i}_{j}.npy')
                np.save(file, array)
                obj[k1][k2] = np.load(file, mmap_mode='r')
        return obj


class RfGlmParamsTemplate(FastStaParamsTemplate):
    """GLM parameters. Adds the GLM hyperparameters to the stimulus and trace preprocessing of the fast STA."""

    @property
    def definition(self):
        definition = super().definition + """
        df_ts : <blob>  # candidate degrees of freedom of the temporal spline basis
        df_ws : <blob>  # candidate degrees of freedom of the spatial spline basis
        betas : <blob>  # candidate regularization strengths
        kfold : int32
        metric = "mse": enum('mse', 'corrcoef')
        output_nonlinearity = 'none' : varchar(63)
        other_params_dict : <blob>
        """
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @staticmethod
    def get_other_params_dict_default(other_params_dict: dict | None = None) -> dict:
        """Return the default `other_params_dict`, updated with `other_params_dict`."""
        params = dict(
            min_iters=100, max_iters=2000, step_size=0.1, step_size_finetune=0.03, tolerance=5,
            alphas=(1.,), verbose=100, n_perm=20, min_cc=0.2, seed=42,
            fit_R=False, fit_intercept=True, init_method='random', atol=1e-5,
            distr='gaussian', frac_test=0,
        )
        params.update(other_params_dict or dict())
        return params

    def add_default(
            self, stim_names: list | None = None, dnoise_params_id: int = 1, fit_kind: str = "gradient",
            fupsample_trace: int = 10, fupsample_stim: int = 4, lowpass_cutoff: float = 0,
            pre_blur_sigma_s: float = 0, post_blur_sigma_s: float = 0,
            filter_dur_s_past: float = 1.2, filter_dur_s_future: float = 0.2,
            df_ts: tuple = (10,), df_ws: tuple = (9,), betas: tuple = (0.005,),
            kfold: int = 0, metric: str = "mse", output_nonlinearity: str = "none",
            other_params_dict: dict | None = None,
            skip_duplicates: bool = False, verbose: bool = True) -> None:
        """Add default GLM parameter set to the table.

        Stimulus and trace preprocessing parameters are described in `FastStaParamsTemplate.add_default`.
        The defaults fit a single model to all data (no hyperparameter search, no test set), which is fast and
        gives reasonable RFs.

        Parameters
        ----------
        df_ts : tuple, optional
            Candidate degrees of freedom of the temporal spline basis. Default is (10,).
        df_ws : tuple, optional
            Candidate degrees of freedom of the spatial spline basis, ignored for 1D stimuli. Default is (9,).
        betas : tuple, optional
            Candidate regularization strengths. Default is (0.005,).
        kfold : int, optional
            Number of cross-validation folds. 0 fits a single model without dev set, which requires a single
            candidate for each of df_ts, df_ws and betas. Default is 0.
        metric : str, optional
            Optimization metric, 'mse' or 'corrcoef'. Default is 'mse'.
        output_nonlinearity : str, optional
            Output nonlinearity of the GLM. Default is 'none'.
        other_params_dict : dict or None, optional
            Updates for the other GLM parameters, see `ReceptiveFieldGLM`. If None, defaults are used.
        """
        super().add_default(
            stim_names=stim_names, dnoise_params_id=dnoise_params_id, fit_kind=fit_kind,
            fupsample_trace=fupsample_trace, fupsample_stim=fupsample_stim, lowpass_cutoff=lowpass_cutoff,
            pre_blur_sigma_s=pre_blur_sigma_s, post_blur_sigma_s=post_blur_sigma_s,
            filter_dur_s_past=filter_dur_s_past, filter_dur_s_future=filter_dur_s_future,
            skip_duplicates=skip_duplicates, verbose=verbose,
            df_ts=tuple(df_ts), df_ws=tuple(df_ws), betas=tuple(betas), kfold=kfold, metric=metric,
            output_nonlinearity=output_nonlinearity,
            other_params_dict=self.get_other_params_dict_default(other_params_dict))


class RfGlmTemplate(FastStaTemplate):
    """GLM receptive fields.

    Subclass of the fast STA template: the design matrix, trace preprocessing, alignment and population logic
    are shared. Only the fit of a single trace is replaced by the GLM fit.
    """
    database = ""

    @property
    def definition(self):
        definition = """
        # Spatio-temporal receptive fields estimated with a GLM
        -> self.params_table
        -> self.traces_table
        ---
        rf: <npy@processed>  # spatio-temporal receptive field, same layout as the fast STA: (time, ..., color)
        model_dict: <blob@processed>
        quality_dict: <blob>
        """
        return definition

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    def prepare_sta_params(self, *restrictions) -> dict:
        """Fetch the STA parameters and add the GLM parameters.

        Parameters
        ----------
        *restrictions : dict
            Optional DataJoint restrictions to select a unique parameter set.

        Returns
        -------
        dict
            STA parameters (see `FastStaTemplate.prepare_sta_params`) and the keys "glm_params" and
            "basis_projections" (the design matrix projected onto the spline bases, shared by all traces).
        """
        sta_params = super().prepare_sta_params(*restrictions)
        names = ('df_ts', 'df_ws', 'betas', 'kfold', 'metric', 'output_nonlinearity', 'other_params_dict')
        values = self._select_params(*restrictions).fetch1(*names)
        sta_params['glm_params'] = dict(zip(names, values))

        model = ReceptiveFieldGLM(
            x_stimulus=sta_params['x_stimulus'], trace=None, **self._get_glm_kwargs(sta_params))
        sta_params['basis_projections'] = _MemmappedArrays.from_arrays(
            model.compute_basis_projections(), prefix='djimaging_glm_')

        # Use the degrees of freedom clipped to the RF dimensions, so that the clipping warns only once here,
        # and not again for every trace.
        sta_params['glm_params'].update(
            df_ts=tuple(int(df) for df in model.df_ts),
            df_ws=() if model.df_ws is None else tuple(int(df) for df in model.df_ws))
        return sta_params

    @staticmethod
    def _get_glm_kwargs(sta_params: dict) -> dict:
        """Arguments of `ReceptiveFieldGLM` except for the data."""
        glm_params = sta_params['glm_params']
        return dict(
            dt=sta_params['dt'], dims=tuple(sta_params['dims']), rf_time=sta_params['rf_time'],
            shift=sta_params['shift'], burn_in=sta_params['burn_in'],
            df_ts=glm_params['df_ts'], df_ws=glm_params['df_ws'], betas=glm_params['betas'],
            kfold=glm_params['kfold'], metric=glm_params['metric'],
            output_nonlinearity=glm_params['output_nonlinearity'], **glm_params['other_params_dict'])

    def _compute_entries(self, traces: list, sta_params: dict, n_jobs: int) -> list:
        """Fit GLM receptive fields to all traces of a presentation, in parallel processes.

        Parameters
        ----------
        traces : list of np.ndarray
            Preprocessed traces aligned to the stimulus, see `_prepare_trace`.
        sta_params : dict
            Parameters from `prepare_sta_params`. Without "basis_projections", every fit projects the design matrix.
        n_jobs : int
            Maximum number of parallel processes, at most the number of available CPU cores.

        Returns
        -------
        list of dict or None
            Entries with "rf", "model_dict" and "quality_dict". None if the trace is not finite.
        """
        glm_kwargs = self._get_glm_kwargs(sta_params)
        basis_projections = sta_params.get('basis_projections')

        def get_fit_kwargs(trace):
            if basis_projections is not None:
                return dict(glm_kwargs=glm_kwargs, basis_projections=basis_projections)
            return dict(glm_kwargs=glm_kwargs, x_stimulus=sta_params['x_stimulus'][:trace.size])

        idxs = []
        for i, trace in enumerate(traces):
            if np.all(np.isfinite(trace)):
                idxs.append(i)
            else:
                print(f"Skipping trace with {np.sum(~np.isfinite(trace))} non-finite values after alignment.")

        n_jobs = max(1, min(n_jobs, len(idxs), os.cpu_count() or 1))
        if n_jobs == 1:
            results = (fit_receptive_field_glm(traces[i], **get_fit_kwargs(traces[i])) for i in idxs)
        else:
            results = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(
                delayed(fit_receptive_field_glm)(traces[i], single_threaded=True, **get_fit_kwargs(traces[i]))
                for i in idxs)

        entries = [None] * len(traces)
        for i, result in zip(idxs, tqdm(results, total=len(idxs), desc='GLM fits', leave=False)):
            entries[i] = self._to_entry(*result, sta_params=sta_params)
        return entries

    @staticmethod
    def _to_entry(rf: np.ndarray, quality_dict: dict, model_dict: dict, sta_params: dict) -> dict:
        """Entry of a single trace from the GLM fit."""
        # Same layout as the fast STA: (time, ..., color)
        if model_dict['n_colors'] > 1:
            rf = np.moveaxis(rf, 0, -1)
        rf = rf.astype(np.float32)
        assert rf.shape == tuple(sta_params['dims']), (rf.shape, sta_params['dims'])

        return dict(rf=rf, model_dict=model_dict, quality_dict=quality_dict)

    def plot1(self, key: dict | None = None, kind: str = 'summary', **kwargs) -> None:
        """Plot the receptive field.

        Parameters
        ----------
        key : dict or None, optional
            DataJoint key to restrict the table. Default is None.
        kind : str, optional
            'summary' (RF, training curves, quality and predictions), or any kind supported by the fast STA
            ('frames', 'video', 'traces'). Default is 'summary'.
        **kwargs
            Additional keyword arguments passed to the specific plot function.
        """
        if kind != 'summary':
            return super().plot1(key=key, kind=kind, **kwargs)

        key = get_primary_key(table=self, key=key)
        rf, quality_dict, model_dict = (self & key).fetch1('rf', 'quality_dict', 'model_dict')
        rf = _rf_to_channels_first(load_array(rf), model_dict)
        plot_rf_summary(rf=rf, quality_dict=quality_dict, model_dict=model_dict,
                        title=' '.join(str(v) for v in key.values()))
        plt.show()


class RfGlmQualityDictTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
            -> self.glm_table
            ---
            corrcoef_train = NULL : float32
            corrcoef_test = NULL : float32
            corrcoef_dev = NULL : float32
            mse_train = NULL : float32
            mse_dev = NULL : float32
            mse_test = NULL : float32
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

    def make(self, key: dict) -> None:
        """Extract quality metrics from the GLM quality_dict and insert as individual columns.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to compute.
        """
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

    def plot(self, *restrictions) -> None:
        """Fetch quality metrics as a DataFrame (no plot produced).

        Parameters
        ----------
        *restrictions : dict
            Optional DataJoint restrictions to apply before fetching.
        """
        df_q = (self & restrictions).to_pandas()


class RfGlmQualityParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        glm_quality_params_id: int32 # unique param set id
        ---
        min_corrcoef : float32
        max_mse : float32
        perm_alpha : float32
        """
        return definition

    def add_default(self, skip_duplicates: bool = False, **update_kw) -> None:
        """Add default quality parameter set to the table.

        Parameters
        ----------
        skip_duplicates : bool, optional
            If True, skip duplicate entries. Default is False.
        **update_kw
            Additional keyword arguments to override default values.
        """
        key = dict(
            glm_quality_params_id=1,
            min_corrcoef=0.1,
            max_mse=1.0,
            perm_alpha=0.001,
        )
        key.update(**update_kw)
        self.insert1(key, skip_duplicates=skip_duplicates)


class RfGlmQualityTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
        # Quality index of GLM receptive fields
        -> self.glm_table
        -> self.params_table
        ---
        rf_glm_qidx : float32
        '''
        return definition

    @property
    def key_source(self):
        try:
            return self.glm_table().proj() * self.params_table().proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def glm_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    def make(self, key: dict) -> None:
        """Compute a composite GLM quality index and insert it.

        Combines permutation test, correlation coefficient, and MSE thresholds
        into a single quality index between 0 and 1.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to compute.
        """
        quality_dict, model_dict = (self.glm_table() & key).fetch1('quality_dict', 'model_dict')
        min_corrcoef, max_mse, perm_alpha = (self.params_table() & key).fetch1(
            'min_corrcoef', 'max_mse', 'perm_alpha')

        rf_glm_qidx_sum = 0.
        rf_glm_qidx_count = 0

        # Without test data (frac_test=0), there is no permutation test and only the train (and dev) scores are used.
        if perm_alpha > 0 and 'score_test' in quality_dict:
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
                (quality_dict.get('corrcoef_test', np.inf) >= min_corrcoef)
            rf_glm_qidx_sum += float(rf_glm_qidx_corrcoeff)
            rf_glm_qidx_count += 1

        if max_mse > 0:
            rf_glm_qidx_mse = \
                (quality_dict['mse_train'] <= max_mse) & \
                (quality_dict.get('mse_dev', -np.inf) <= max_mse) & \
                (quality_dict.get('mse_test', -np.inf) <= max_mse)
            rf_glm_qidx_sum += float(rf_glm_qidx_mse)
            rf_glm_qidx_count += 1

        if rf_glm_qidx_count == 0:
            raise ValueError(f"No quality criterion applies to {key}: set min_corrcoef or max_mse > 0, "
                             f"or perm_alpha > 0 for GLMs with test data.")

        rf_glm_qidx = rf_glm_qidx_sum / rf_glm_qidx_count

        q_key = key.copy()
        q_key['rf_glm_qidx'] = rf_glm_qidx

        self.insert1(q_key)

    def plot(self, glm_quality_params_id: int = 1) -> None:
        """Plot histogram of GLM quality indices for a given parameter set.

        Parameters
        ----------
        glm_quality_params_id : int, optional
            ID of the quality parameter set to plot. Default is 1.
        """
        rf_glm_qidx = (self & f"glm_quality_params_id={glm_quality_params_id}").to_arrays("rf_glm_qidx")

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.hist(rf_glm_qidx)
        ax.set(title=f"glm_quality_params_id={glm_quality_params_id}", ylabel="rf_glm_qidx")
        ax.set_xlim(-0.1, 1.1)
        plt.show()
