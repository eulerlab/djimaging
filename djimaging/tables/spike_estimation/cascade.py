import warnings

import os
import sys
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils import math_utils, plot_utils, trace_utils, filter_utils

from djimaging.utils.dj_utils import get_primary_key

from djimaging.tables.core.preprocesstraces import detrend_trace, drop_left_and_right, non_negative_trace
from djimaging.utils.plot_utils import plot_trace_and_trigger
from djimaging.utils.trace_utils import get_mean_dt


class CascadeTracesParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        cascadetraces_params_id:    int       # unique param set id
        ---
        window_length:       int       # window length for SavGol filter in seconds
        poly_order:          int       # order of polynomial for savgol filter
        fs_resample = 0 :    float     # Resampling frequency, only applied when > 0.
        """
        return definition

    def add_default(self, cascadetraces_params_id=1, window_length=60, poly_order=3, fs_resample=None,
                    skip_duplicates=False):
        key = dict(
            cascadetraces_params_id=cascadetraces_params_id,
            window_length=window_length,
            poly_order=int(poly_order),
            fs_resample=fs_resample if fs_resample is not None else 0,
        )
        """Add default preprocess parameter to table"""
        self.insert1(key, skip_duplicates=skip_duplicates)


class CascadeTracesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.traces_table
        -> self.cascadetraces_params_table
        ---
        cascade_trace_times: longblob   # Time of preprocessed trace, if not resampled same as in trace.
        cascade_trace:      longblob    # preprocessed trace
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
            return (self.traces_table() & 'trace_flag=1' & 'trigger_flag=1').proj() * \
                self.cascadetraces_params_table().proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        window_len_seconds, poly_order, fs_resample = \
            (self.cascadetraces_params_table() & key).fetch1('window_length', 'poly_order', 'fs_resample')

        trace_times, trace = (self.traces_table() & key).fetch1('trace_times', 'trace')
        stim_start = (self.presentation_table() & key).fetch1('triggertimes')[0]

        trace = np.asarray(trace).copy()
        # get fs
        dt, dt_rel_error = get_mean_dt(tracetime=trace_times)
        fs = 1. / dt

        if dt_rel_error > 0.1:
            warnings.warn('Inconsistent step-sizes in trace, resample trace.')
            tracetime, trace = filter_utils.resample_trace(tracetime=trace_times, trace=trace, dt=dt)

        trace = drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3), inplace=True)
        trace, smoothed_trace = detrend_trace(trace, window_len_seconds, fs, poly_order)

        if stim_start is not None:
            if not np.any(trace_times < stim_start):
                raise ValueError(f"stim_start={stim_start:.1g}, trace_start={trace_times.min():.1g}")

        trace = non_negative_trace(trace, inplace=True)
        trace /= np.std(trace)

        self.insert1(dict(key, cascade_trace_times=trace_times, cascade_trace=trace))

    def plot1(self, key=None, xlim=None, ylim=None):
        key = get_primary_key(self, key)

        cascade_trace_times, cascade_trace = (self & key).fetch1(
            "cascade_trace_times", "cascade_trace")
        trace_times, trace = (self.traces_table() & key).fetch1("trace_times", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex='all')

        ax = axs[0]
        plot_trace_and_trigger(time=trace_times, trace=trace,
                               triggertimes=triggertimes, ax=ax, title=str(key))
        ax.set(ylabel='mean subtracted\nraw trace')

        ax = axs[1]
        plot_trace_and_trigger(time=cascade_trace_times, trace=cascade_trace, triggertimes=triggertimes, ax=ax)
        ax.set(ylabel='preprocessed\ntrace')

        ax.set(xlim=xlim, ylim=ylim)

        plt.show()

    def plot(self, restriction=None, sort=True):
        if restriction is None:
            restriction = dict()

        cascade_traces = (self & restriction).fetch("cascade_trace")
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
        cascade_params_id: int # unique param set id
        ---
        model_name : varchar(255)
        cascade_model_subfolder : varchar(255)
        cascade_folder : varchar(255)
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
        spike_prob_times: longblob
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
        assert make_kwargs is None, "make_kwargs is not supported for this table"
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
                make_kwargs=dict(
                    cascade_models_path=os.path.join(cascade_folder, cascade_model_subfolder),
                    cascade=cascade),
            )

    def make(self, key, cascade_models_path, cascade):
        model_name = (self.cascade_params_table & key).fetch1('model_name')
        trace_times, traces, roi_ids = (self.cascadetraces_table & key).fetch(
            'cascade_trace_times', 'cascade_trace', 'roi_id')

        spike_probs = cascade.predict(model_name, np.stack(traces), verbosity=1, model_folder=cascade_models_path)

        for trace_time, spike_prob, roi_id in zip(trace_times, spike_probs, roi_ids):
            self.insert1(dict(key, spike_prob_times=trace_time, spike_prob=spike_prob, roi_id=roi_id))

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(self, key)

        spike_prob_times, spike_prob = (self & key).fetch1("spike_prob_times", "spike_prob")
        trace_times, traces = (self.cascadetraces_table & key).fetch1('cascade_trace_times', 'cascade_trace')

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(trace_times, traces, 'k', label='trace')
        ax.legend(loc='upper left')
        ax = ax.twinx()
        ax.plot(spike_prob_times, spike_prob, 'r', label='spike_prob')
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
