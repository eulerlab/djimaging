from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import PlaceholderTable, get_plot_key
from djimaging.utils.rf_utils import get_sets, rfest


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

        rf, rf_pred, X, y, dt = compute_receptive_field(
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
        key = get_plot_key(table=self, key=key)

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


def compute_receptive_field(trace, tracetime, stim, stimtime, frac_train, frac_dev, dur_filter_s,
                            kind='sta', fupsample=1, gradient=False, norm_stim=True, norm_trace=True):
    assert trace.ndim == 1
    assert tracetime.ndim == 1
    assert trace.size == tracetime.size
    assert stim.shape[0] == stimtime.shape[0], f"{stim.shape} vs. {stimtime.shape}"

    kind = kind.lower()

    X, y, dt = get_sets(
        stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime,
        frac_train=frac_train, frac_dev=frac_dev, fupsample=fupsample, gradient=gradient,
        norm_stim=norm_stim, norm_trace=norm_trace)

    if kind in ['sta', 'mle']:
        assert 'dev' not in X and 'dev' not in y, 'Development sets are not use for these rfs'
        rf, rf_pred = fit_sta(X, y, dur_filter_s, dt, kind=kind)
    else:
        raise NotImplementedError(f"kind={kind}")

    return rf, rf_pred, X, y, dt


def fit_sta(X, y, dur_filter_s, dt, kind='sta'):
    """Compute STA or MLE"""
    assert rfest is not None
    from rfest.GLM._base import Base as BaseModel

    kind = kind.lower()
    assert kind in ['sta', 'mle'], kind

    dim_t = int(np.ceil(dur_filter_s / dt))
    dims = (dim_t,) + X['train'].shape[1:]

    burn_in = dims[0] - 1

    X_train_dm = rfest.utils.build_design_matrix(X['train'], dims[0])[burn_in:]
    y_train_dm = y['train'][burn_in:]

    model = BaseModel(X=X_train_dm, y=y_train_dm, dims=dims, compute_mle=kind == 'mle')
    rf = model.w_sta if kind == 'sta' else model.w_mle

    rf_pred = dict()
    rf_pred['burn_in'] = burn_in
    y_pred_train = X_train_dm @ rf
    rf_pred['y_pred_train'] = y_pred_train
    rf_pred['cc_train'] = np.corrcoef(y['train'][burn_in:], y_pred_train)[0, 1]
    rf_pred['mse_train'] = np.mean((y['train'][burn_in:] - y_pred_train) ** 2)

    if 'test' in X:
        X_test_dm = rfest.utils.build_design_matrix(X['test'], dims[0])[burn_in:]
        y_pred_test = X_test_dm @ rf
        rf_pred['y_pred_test'] = y_pred_test
        rf_pred['cc_test'] = np.corrcoef(y['test'][burn_in:], y_pred_test)[0, 1]
        rf_pred['mse_test'] = np.mean((y['test'][burn_in:] - y_pred_test) ** 2)

    return rf.reshape(dims), rf_pred
