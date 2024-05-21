"""
Tables for spike-triggered averages (STA) receptive fields (RFs).
Actually it's implemented for calcium event triggered averages, as we usually have calcium traces.

Example usage:

from djimaging.tables import receptivefield

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
class STAParams(receptivefield.STAParamsTemplate):
    pass


@schema
class STA(receptivefield.STATemplate):
    noise_traces_table = DNoiseTrace
    params_table = STAParams

    class DataSet(receptivefield.STATemplate.DataSet):
        pass


@schema
class SplitRFParams(receptivefield.SplitRFParamsTemplate):
    pass


@schema
class SplitRF(receptivefield.SplitRFTemplate):
    rf_table = STA
    split_rf_params_table = SplitRFParams


@schema
class FitGauss2DRF(receptivefield.FitGauss2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


@schema
class FitDoG2DRF(receptivefield.FitDoG2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus

@schema
class TempRFProperties(receptivefield.TempRFPropertiesTemplate):
    _max_dt_future = 0.1
    split_rf_table = SplitRF
    rf_table = STA
"""

from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.receptive_fields.fit_rf_utils import compute_linear_rf
from djimaging.utils.receptive_fields.plot_rf_utils import plot_rf_frames, plot_rf_video


class STAParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        sta_params_id: tinyint unsigned # unique param set id
        ---
        rf_method : enum("sta", "mle")
        filter_dur_s_past : float # filter duration in seconds into the past
        filter_dur_s_future : float # filter duration in seconds into the future
        frac_train : float  # Fraction of data used for training in (0, 1].
        frac_dev : float  # Fraction of data used for hyperparameter optimization in [0, 1).
        frac_test : float  # Fraction of data used for testing [0, 1).
        store_x : enum("shape", "data")  # Store x (stimulus) as data or shape (less storage)?
        store_y : enum("shape", "data")  # Store y (response) as data or shape (less storage)?
        """
        return definition

    def add_default(
            self, sta_params_id=1, rf_method="sta", filter_dur_s_past=1., filter_dur_s_future=0.,
            frac_train=0.8, frac_dev=0., frac_test=0.2, store_x='shape', store_y='shape',
            skip_duplicates=False):
        """Add default preprocess parameter to table"""

        if store_x == 'data' or store_y == 'data':
            if input("Are you sure you want to store the data? This creates a lot of overhead. (y/n) ") != 'y':
                return

        key = dict(
            sta_params_id=sta_params_id,
            rf_method=rf_method, filter_dur_s_past=filter_dur_s_past, filter_dur_s_future=filter_dur_s_future,
            frac_train=frac_train, frac_dev=frac_dev, frac_test=frac_test,
            store_x=store_x, store_y=store_y,
        )

        self.insert1(key, skip_duplicates=skip_duplicates)


class STATemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.noise_traces_table
        -> self.params_table
        ---
        rf: longblob  # spatio-temporal receptive field
        rf_time: longblob #  time of RF, depends on dt and shift
        dt: float  # Time step between frames
        shift: int  # Shift of stimulus relative to trace. If negative, prediction looks into future.
        '''
        return definition

    @property
    def key_source(self):
        try:
            return (self.params_table * self.noise_traces_table).proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def noise_traces_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    class DataSet(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            kind : enum('train', 'dev', 'test')  # Data set kind
            ---
            x : longblob  # Input
            y : longblob  # Output
            burn_in : int unsigned  # Burned output s.t. y_pred.size + burn_in == y.size
            y_pred : longblob # predicted output
            cc : float  # Correlation
            mse : float  # Mean Squared Error
            """
            return definition

    def make(self, key, verbose=False):
        # TODO: Fix overhead by precomputing the design matrix

        filter_dur_s_past, filter_dur_s_future, rf_method = (self.params_table() & key).fetch1(
            "filter_dur_s_past", "filter_dur_s_future", "rf_method")
        store_x, store_y = (self.params_table() & key).fetch1("store_x", "store_y")
        frac_train, frac_dev, frac_test = (self.params_table() & key).fetch1("frac_train", "frac_dev", "frac_test")
        assert np.isclose(frac_train + frac_dev + frac_test, 1.0)

        noise_dt, trace, stim_idxs = (self.noise_traces_table() & key).fetch1('noise_dt', 'trace', 'stim_idxs')
        stim = (self.noise_traces_table.stimulus_table() & key).fetch1("stim_trace")
        stim = stim[stim_idxs].astype(trace.dtype)

        rf, rf_time, rf_pred, x, y, shift = compute_linear_rf(
            dt=noise_dt, trace=trace, stim=stim, frac_train=frac_train, frac_dev=frac_dev, kind=rf_method,
            filter_dur_s_past=filter_dur_s_past, filter_dur_s_future=filter_dur_s_future,
            threshold_pred=np.all(trace >= 0), batch_size_n=6_000_000, verbose=verbose)

        rf_key = deepcopy(key)
        rf_key['rf'] = rf.astype(np.float32)
        rf_key['rf_time'] = rf_time.astype(np.float32)
        rf_key['dt'] = noise_dt
        rf_key['shift'] = shift
        self.insert1(rf_key)

        for k in x.keys():
            rf_dataset_key = deepcopy(key)
            rf_dataset_key['kind'] = k
            rf_dataset_key['burn_in'] = rf_pred['burn_in']
            rf_dataset_key['x'] = x[k].astype(np.float32) if store_x == 'data' else x[k].shape
            rf_dataset_key['y'] = y[k].astype(np.float32) if store_y == 'data' else y[k].shape
            rf_dataset_key['y_pred'] = rf_pred[f'y_pred_{k}'].astype(np.float32) \
                if store_y == 'data' else rf_pred[f'y_pred_{k}'].shape
            rf_dataset_key['cc'] = rf_pred[f'cc_{k}']
            rf_dataset_key['mse'] = rf_pred[f'mse_{k}']
            self.DataSet().insert1(rf_dataset_key)

    def plot1_traces(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)

        from matplotlib import pyplot as plt

        data = (self * self.DataSet() & key).fetch()

        fig, axs = plt.subplots(len(data), 1, figsize=(10, 3 * len(data)), squeeze=False)
        axs = axs.flat

        for ax, row in zip(axs, data):
            ax.set(title=f"{row['kind']}   cc={row['cc']:.2f}   mse={row['mse']:.2f}", ylabel='y')
            time = np.arange(row['y'].size) * row['dt']

            burn_in = row['burn_in']

            ax.plot(time[:burn_in + 1], row['y'][:burn_in + 1], label='_', ls='--', c='C0')
            ax.plot(time[burn_in:], row['y'][burn_in:], label='data', c='C0')
            ax.plot(time[burn_in:], row['y_pred'], label='pred', alpha=0.8, c='C1')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_xlim(xlim)

        axs[-1].set(xlabel='Time')
        plt.tight_layout()

    def plot1_frames(self, key=None, downsample=1):
        key = get_primary_key(table=self, key=key)
        rf, rf_time = (self & key).fetch1('rf', 'rf_time')
        plot_rf_frames(rf, rf_time, downsample=downsample)

    def plot1_video(self, key=None, fps=10):
        key = get_primary_key(table=self, key=key)
        rf, rf_time = (self & key).fetch1('rf', 'rf_time')
        return plot_rf_video(rf, rf_time, fps=fps)
