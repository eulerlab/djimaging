from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import PlaceholderTable, get_primary_key
from djimaging.tables.receptivefield.rf_utils import compute_sta


class STAParamsTemplate(dj.Lookup):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        rfparams_id: int # unique param set id
        ---
        rf_method : enum("sta", "mle")
        dur_filter_s : float # minimum duration of filter in seconds
        trace_gradient : tinyint unsigned  # Use gradient of signal instead of signal?
        trace_fupsample : tinyint unsigned  # Upsample trace by integer factor?
        norm_stim : tinyint unsigned  # Normalize (znorm) stimulus?
        norm_trace : tinyint unsigned  # Normalize (znorm) trace?
        frac_train : float  # Fraction of data used for training in (0, 1].
        frac_dev : float  # Fraction of data used for hyperparameter optimization in [0, 1).
        frac_test : float  # Fraction of data used for testing [0, 1).
        store_x : enum("data", "shape")  # Store x (stimulus) as data or shape (less storage)?
        store_y : enum("data", "shape")  # Store y (response) as data or shape (less storage)?
        """
        return definition

    def add_default(self, skip_duplicates=False):
        """Add default preprocess parameter to table"""
        key = {
            'rfparams_id': 1,
            'rf_method': "sta",
            'dur_filter_s': 1.0,
            'norm_trace': 1,
            'norm_stim': 1,
            'trace_gradient': 0,
            'trace_fupsample': 1,
            'frac_train': 0.8,
            'frac_dev': 0.,
            'frac_test': 0.2,
            'store_x': 'data',
            'store_y': 'data',
        }
        self.insert1(key, skip_duplicates=skip_duplicates)


class STATemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.preprocesstraces_table
        -> self.params_table
        ---
        rf: longblob  # spatio-temporal receptive field
        dt : float  # Time-step of time component
        '''
        return definition

    stimulus_table = PlaceholderTable
    presentation_table = PlaceholderTable
    traces_table = PlaceholderTable
    preprocesstraces_table = PlaceholderTable
    params_table = PlaceholderTable

    @property
    def key_source(self):
        try:
            return self.params_table() * self.preprocesstraces_table() & \
                   (self.stimulus_table() & "stim_family = 'noise'")
        except TypeError:
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

    def make(self, key):
        stim = (self.stimulus_table() & key).fetch1("stim_trace")
        stimtime = (self.presentation_table() & key).fetch1('triggertimes')

        assert stim is not None, "stim_trace in stimulus table must be set."

        trace, tracetime = (self.preprocesstraces_table() & key).fetch1('preprocess_trace', 'preprocess_trace_times')

        frac_train, frac_dev, frac_test, norm_stim, norm_trace, dur_filter_s, store_x, store_y = \
            (self.params_table() & key).fetch1(
                "frac_train", "frac_dev", "frac_test", "norm_stim", "norm_trace", "dur_filter_s", "store_x", "store_y")

        assert np.isclose(frac_train + frac_dev + frac_test, 1.0)

        rf, rf_pred, X, y, dt = compute_sta(
            trace=trace, tracetime=tracetime, stim=stim, stimtime=stimtime,
            frac_train=frac_train, frac_dev=frac_dev, dur_filter_s=dur_filter_s,
            norm_stim=norm_stim, norm_trace=norm_trace)

        rf_key = deepcopy(key)
        rf_key['rf'] = rf
        rf_key['dt'] = dt
        self.insert1(rf_key)

        for k in X.keys():
            rf_dataset_key = deepcopy(key)
            rf_dataset_key['kind'] = k
            rf_dataset_key['burn_in'] = rf_pred['burn_in']
            rf_dataset_key['x'] = X[k] if store_x == 'data' else X[k].shape
            rf_dataset_key['y'] = y[k] if store_y == 'data' else y[k].shape
            rf_dataset_key['y_pred'] = rf_pred[f'y_pred_{k}'] if store_y == 'data' else rf_pred[f'y_pred_{k}'].shape
            rf_dataset_key['cc'] = rf_pred[f'cc_{k}']
            rf_dataset_key['mse'] = rf_pred[f'mse_{k}']
            self.DataSet().insert1(rf_dataset_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        from matplotlib import pyplot as plt

        data = (self * self.DataSet() & key).fetch()

        fig, axs = plt.subplots(len(data), 1, figsize=(10, 3 * len(data)))

        for ax, row in zip(axs, data):
            ax.set(title=f"{row['kind']}   cc={row['cc']:.2f}   mse={row['mse']:.2f}", ylabel='y')
            time = np.arange(row['y'].size) * row['dt']

            burn_in = row['burn_in']

            ax.plot(time[:burn_in + 1], row['y'][:burn_in + 1], label='_', ls='--', c='C0')
            ax.plot(time[burn_in:], row['y'][burn_in:], label='data', c='C0')
            ax.plot(time[burn_in:], row['y_pred'], label='pred', alpha=0.8, c='C1')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        axs[-1].set(xlabel='Time')
        plt.tight_layout()


