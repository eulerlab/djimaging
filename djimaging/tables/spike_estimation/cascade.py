"""
Tables for spike estimation using the Cascade toolbox.

Example usage:

from djimaging.tables import spike_estimation

@schema
class CascadeTraceParams(spike_estimation.CascadeTracesParamsTemplate):
    pass

@schema
class CascadeTraces(spike_estimation.CascadeTracesTemplate):
    cascadetraces_params_table = CascadeTraceParams
    presentation_table = Presentation
    traces_table = Traces

@schema
class CascadeParams(spike_estimation.CascadeParamsTemplate):
    pass

@schema
class CascadeSpikes(spike_estimation.CascadeSpikesTemplate):
    presentation_table = Presentation
    cascadetraces_params_table = CascadeTraceParams
    cascadetraces_table = CascadeTraces
    cascade_params_table = CascadeParams
"""

import warnings

import os
import sys
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils import math_utils, plot_utils, trace_utils, filter_utils

from djimaging.utils.dj_utils import get_primary_key

from djimaging.tables.core.preprocesstraces import detrend_trace, drop_left_and_right
from djimaging.utils.plot_utils import plot_trace_and_trigger
from djimaging.utils.trace_utils import get_mean_dt


class CascadeTracesParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.stimulus_table
        cas_params_id:    int       # unique param set id
        ---
        window_length:       int       # window length for SavGol filter in seconds
        poly_order:          int       # order of polynomial for savgol filter
        q_lower:             float     # lower percentile for dF/F computation
        q_upper:             float     # upper percentile for dF/F computation
        f_cutoff:            float     # cutoff frequency for lowpass filter, only applied when > 0.
        fs_resample = 0 :    float     # Resampling frequency, only applied when > 0.
        """
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def add_default(self, stim_names=None, cas_params_id=1, window_length=60, poly_order=3,
                    q_lower=2.5, q_upper=80., f_cutoff=None, fs_resample=None, skip_duplicates=False):
        assert q_lower < q_upper, f'q_lower={q_lower} must be smaller than q_upper={q_upper}'
        assert 0 <= q_lower <= 100, f'q_lower={q_lower} must be between 0 and 100'
        assert 0 <= q_upper <= 100, f'q_upper={q_upper} must be between 0 and 100'

        if stim_names is None:
            stim_names = (self.stimulus_table()).fetch('stim_name')

        key = dict(
            cas_params_id=cas_params_id,
            window_length=window_length,
            poly_order=int(poly_order),
            q_lower=q_lower, q_upper=q_upper,
            fs_resample=fs_resample if fs_resample is not None else 0,
            f_cutoff=f_cutoff if f_cutoff is not None else 0,
        )

        for stim_name in stim_names:
            """Add default preprocess parameter to table"""
            stim_key = key.copy()
            stim_key['stim_name'] = stim_name
            self.insert1(stim_key, skip_duplicates=skip_duplicates)


class CascadeTracesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.traces_table
        -> self.cascadetraces_params_table
        ---
        pp_trace:      longblob    # preprocessed trace
        pp_trace_t0:   float      # start time of trace
        pp_trace_dt:   float      # time step of trace
        """
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    @abstractmethod
    def cascadetraces_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.traces_table() & 'trace_valid=1' & 'trigger_valid=1').proj() * \
                self.cascadetraces_params_table().proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        window_len_seconds, poly_order, q_lower, q_upper, f_cutoff, fs_resample = \
            (self.cascadetraces_params_table() & key).fetch1(
                'window_length', 'poly_order', 'q_lower', 'q_upper', 'f_cutoff', 'fs_resample')

        if fs_resample > 0:
            warnings.warn("fs_resample is not implemented yet.")

        trace_t0, trace_dt, trace = (self.traces_table() & key).fetch1('trace_t0', 'trace_dt', 'trace')
        stim_start = (self.presentation_table() & key).fetch1('triggertimes')[0]

        if stim_start is not None:
            if stim_start < trace_t0:
                raise ValueError(f"stim_start={stim_start:.1g}, trace_start={trace_t0:.1g}")

        pp_trace = compute_cascade_firing_rate(
            trace, trace_dt=trace_dt, window_len_seconds=window_len_seconds, poly_order=poly_order,
            q_lower=q_lower, q_upper=q_upper, f_cutoff=f_cutoff)

        self.insert1(dict(key, pp_trace=pp_trace.astype(np.float32), pp_trace_t0=trace_t0, pp_trace_dt=trace_dt))

    def plot1(self, key=None, xlim=None, ylim=None):
        key = get_primary_key(self, key)

        pp_trace_t0, pp_trace_dt, pp_trace = (self & key).fetch1("pp_trace_t0", "pp_trace_dt", "pp_trace")
        trace_t0, trace_dt, trace = (self.traces_table() & key).fetch1("trace_t0", "trace_dt", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        trace_times = np.arange(len(trace)) * trace_dt + trace_t0
        rate_times = np.arange(len(pp_trace)) * pp_trace_dt + pp_trace_t0

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex='all')

        ax = axs[0]
        plot_trace_and_trigger(time=trace_times, trace=trace, triggertimes=triggertimes, ax=ax, title=str(key))
        ax.set(ylabel='mean subtracted\nraw trace')

        ax = axs[1]
        plot_trace_and_trigger(time=rate_times, trace=pp_trace, triggertimes=triggertimes, ax=ax)
        ax.set(ylabel='preprocessed\ntrace')

        ax.set(xlim=xlim, ylim=ylim)

        plt.show()

    def plot(self, restriction=None, sort=True):
        if restriction is None:
            restriction = dict()

        cascade_traces = (self & restriction).fetch("pp_trace")
        cascade_traces = math_utils.padded_vstack(cascade_traces, cval=np.nan)
        n = cascade_traces.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 1 + np.minimum(n * 0.1, 10)))
        if len(restriction) > 0:
            plot_utils.set_long_title(fig=fig, title=restriction)

        sort_idxs = trace_utils.argsort_traces(cascade_traces, ignore_nan=True) if sort else np.arange(n)

        ax.set_title('cascade_traces')
        plot_utils.plot_signals_heatmap(ax=ax, signals=cascade_traces[sort_idxs, :], symmetric=False)
        plt.show()


class CascadeParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        cascade_params_id: tinyint unsigned # unique param set id
        ---
        model_name : varchar(191)
        cascade_model_subfolder : varchar(191)
        cascade_folder : varchar(191)
        """
        return definition

    def add_default(self, model_name, cascade_params_id=1,
                    cascade_folder='/gpfs01/euler/data/Resources/GitHub/external_repos/Cascade/',
                    cascade_model_subfolder='Pretrained_models', skip_duplicates=False):
        """Add default preprocess parameter to table"""

        key = dict(
            cascade_params_id=cascade_params_id,
            model_name=model_name,
            cascade_folder=cascade_folder,
            cascade_model_subfolder=cascade_model_subfolder,
        )

        self.insert1(key, skip_duplicates=skip_duplicates)


class CascadeSpikesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.cascadetraces_table 
        -> self.cascade_params_table
        ---
        spike_prob:       longblob
        """
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def cascadetraces_params_table(self):
        pass

    @property
    @abstractmethod
    def cascadetraces_table(self):
        pass

    @property
    @abstractmethod
    def cascade_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.presentation_table().proj() * \
                self.cascade_params_table().proj() * \
                self.cascadetraces_params_table.proj()
        except (AttributeError, TypeError):
            pass

    def populate(
            self,
            *restrictions,
            suppress_errors=False,
            return_exception_objects=False,
            reserve_jobs=False,
            order="original",
            limit=None,
            max_calls=None,
            display_progress=False,
            processes=1,
            make_kwargs=None,
    ):
        if len(restrictions) == 0:
            cascade_params_ids = self.cascade_params_table.fetch('cascade_params_id')
        else:
            cascade_params_ids = (self.cascade_params_table & restrictions).fetch('cascade_params_id')

        for cascade_params_id in cascade_params_ids:
            cascade_folder, cascade_model_subfolder = (
                    self.cascade_params_table & dict(cascade_params_id=cascade_params_id)).fetch1(
                'cascade_folder', 'cascade_model_subfolder')

            if cascade_folder not in sys.path:
                assert os.path.isdir(cascade_folder)
                print(f'Adding {cascade_folder} to sys.path')
                sys.path = [cascade_folder] + sys.path

            try:
                from cascade2p import checks
                from cascade2p import cascade
            except ImportError:
                raise ImportError(f'Failed to import cascade from folder {cascade_folder}')

            checks.check_packages()

            if make_kwargs is None:
                make_kwargs = dict()
            else:
                if 'cascade' in make_kwargs:
                    raise ValueError("cascade cannot be passed as make_kwargs.")
                if 'cascade_models_path' in make_kwargs:
                    raise ValueError("cascade_models_path cannot be passed as make_kwargs.")

            make_kwargs['cascade'] = cascade
            make_kwargs['cascade_models_path'] = os.path.join(cascade_folder, cascade_model_subfolder)

            super().populate(
                *restrictions,
                dict(cascade_params_id=cascade_params_id),
                suppress_errors=suppress_errors,
                return_exception_objects=return_exception_objects,
                reserve_jobs=reserve_jobs,
                order=order,
                limit=limit,
                max_calls=max_calls,
                display_progress=display_progress,
                processes=processes,
                make_kwargs=make_kwargs,
            )

    def make(self, key, cascade_models_path, cascade, verboselvl=0):
        model_name = (self.cascade_params_table & key).fetch1('model_name')
        pp_traces, roi_ids = (self.cascadetraces_table & key).fetch('pp_trace', 'roi_id')

        if len(pp_traces) == 0:
            return
        elif len(pp_traces) > 1:
            pp_traces = np.stack(pp_traces)

        spike_probs = cascade.predict(model_name, pp_traces, verbosity=verboselvl, model_folder=cascade_models_path)

        for spike_prob, roi_id in zip(spike_probs, roi_ids):
            self.insert1(dict(key, spike_prob=spike_prob.astype(np.float32), roi_id=roi_id))

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(self, key)

        spike_prob = (self & key).fetch1("spike_prob")
        pp_trace_t0, pp_trace_dt, pp_trace = (self.cascadetraces_table & key).fetch1(
            "pp_trace_t0", "pp_trace_dt", "pp_trace")

        trace_times = np.arange(len(pp_trace)) * pp_trace_dt + pp_trace_t0

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(trace_times, pp_trace, 'k', label='trace')
        ax.legend(loc='upper left')
        ax = ax.twinx()
        ax.plot(trace_times, spike_prob, 'r', label='spike_prob')
        ax.legend(loc='upper right')
        ax.set(xlim=xlim)
        plt.show()

    def plot(self, restriction=None, sort=True):
        if restriction is None:
            restriction = dict()

        spike_prob = (self & restriction).fetch("spike_prob")
        spike_prob = math_utils.padded_vstack(spike_prob, cval=np.nan)
        n = spike_prob.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 1 + np.minimum(n * 0.1, 10)))
        if len(restriction) > 0:
            plot_utils.set_long_title(fig=fig, title=restriction)

        sort_idxs = trace_utils.argsort_traces(spike_prob, ignore_nan=True) if sort else np.arange(n)

        ax.set_title('spike_prob')
        plot_utils.plot_signals_heatmap(ax=ax, signals=spike_prob[sort_idxs, :], symmetric=False)
        plt.show()


def compute_cascade_firing_rate(trace, trace_dt, window_len_seconds, poly_order, q_lower=2.5, q_upper=80,
                                f_cutoff=None):
    """Preprocess trace for cascade toolbox. Includes detrending, lowpass filtering and dF/F computation."""
    trace = np.asarray(trace).copy()

    trace = drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3), inplace=True)
    trace, smoothed_trace = detrend_trace(trace, window_len_seconds, 1. / trace_dt, poly_order)

    if (f_cutoff is not None) and (f_cutoff > 0):
        trace = filter_utils.lowpass_filter_trace(trace=trace, fs=1. / trace_dt, f_cutoff=f_cutoff)

    trace = compute_dff(trace, q_lower=q_lower, q_upper=q_upper)

    return trace


def compute_dff(trace, q_lower=2.5, q_upper=80.):
    """Compute dF/F proxy from trace."""
    trace = np.asarray(trace).copy()

    trace_lower = np.percentile(trace, q_lower)
    trace_upper = np.percentile(trace, q_upper)

    # Compute the change in fluorescence (ΔF)
    delta_fluorescence = trace - trace_lower

    # Compute ΔF/F
    if trace_upper - trace_lower > 0.:
        trace = delta_fluorescence / float(trace_upper - trace_lower)

    return trace
