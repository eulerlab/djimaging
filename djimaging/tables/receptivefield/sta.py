from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.tables.receptivefield.rf_utils import compute_linear_rf
from djimaging.utils.dj_utils import get_primary_key


class STAParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        sta_params_id: int # unique param set id
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
        rf_time: longblob #  time of RF, depends from dt and shift
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
        filter_dur_s_past, filter_dur_s_future, rf_method = (self.params_table() & key).fetch1(
            "filter_dur_s_past", "filter_dur_s_future", "rf_method")
        store_x, store_y = (self.params_table() & key).fetch1("store_x", "store_y")
        frac_train, frac_dev, frac_test = (self.params_table() & key).fetch1("frac_train", "frac_dev", "frac_test")
        dt, time, trace, stim = (self.noise_traces_table() & key).fetch1('dt', 'time', 'trace', 'stim')
        assert np.isclose(frac_train + frac_dev + frac_test, 1.0)

        rf, rf_time, rf_pred, x, y, shift = compute_linear_rf(
            dt=dt, trace=trace, stim=stim, frac_train=frac_train, frac_dev=frac_dev, kind=rf_method,
            filter_dur_s_past=filter_dur_s_past, filter_dur_s_future=filter_dur_s_future,
            threshold_pred=np.all(trace >= 0), batch_size_n=6_000_000, verbose=verbose)

        rf_key = deepcopy(key)
        rf_key['rf'] = rf
        rf_key['rf_time'] = rf_time
        rf_key['dt'] = dt
        rf_key['shift'] = shift
        self.insert1(rf_key)

        for k in x.keys():
            rf_dataset_key = deepcopy(key)
            rf_dataset_key['kind'] = k
            rf_dataset_key['burn_in'] = rf_pred['burn_in']
            rf_dataset_key['x'] = x[k] if store_x == 'data' else x[k].shape
            rf_dataset_key['y'] = y[k] if store_y == 'data' else y[k].shape
            rf_dataset_key['y_pred'] = rf_pred[f'y_pred_{k}'] if store_y == 'data' else rf_pred[f'y_pred_{k}'].shape
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

    def plot1_frames(self, key=None):
        key = get_primary_key(table=self, key=key)

        from matplotlib import pyplot as plt
        from djimaging.utils.plot_utils import plot_srf

        rf, rf_time = (self & key).fetch1('rf', 'rf_time')

        if rf.ndim == 2:
            fig, axs = plt.subplots(1, 1, figsize=(10, 3))
            axs.plot(rf_time, rf)
        elif rf.ndim == 3:
            n_rows = int(np.ceil(np.sqrt(rf.shape[0])))
            n_cols = int(np.ceil(rf.shape[0] / n_rows))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False, sharey='all',
                                    sharex='all')
            axs = axs.flat
            vabsmax = np.max(np.abs(rf))
            for ax, frame in zip(axs, rf):
                plot_srf(frame, ax=ax, vabsmax=vabsmax)
        else:
            raise NotImplementedError(rf.shape)

        return fig, axs
