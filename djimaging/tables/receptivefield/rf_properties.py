import warnings
from copy import deepcopy

import datajoint as dj
import numpy as np
from astropy.modeling.fitting import SLSQPLSQFitter
from astropy.modeling.models import Gaussian2D
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import PlaceholderTable, get_primary_key
from djimaging.utils.plot_utils import plot_srf, plot_trf, plot_signals_heatmap
from djimaging.utils.rf_utils import compute_explained_rf, resize_srf, split_strf, \
    compute_polarity_and_peak_idxs, merge_strf


class SplitRFParamsTemplate(dj.Lookup):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        split_rf_params_id: int # unique param set id
        ---
        blur_std : float
        blur_npix : int unsigned
        upsample_srf_scale : int unsigned
        peak_nstd : float  # How many standard deviations does a peak need to be considered peak?
        """
        return definition

    def add_default(self, skip_duplicates=False):
        """Add default preprocess parameter to table"""
        key = {
            'split_rf_params_id': 1,
            'blur_std': 1.,
            'blur_npix': 1,
            'upsample_srf_scale': 0,
            'peak_nstd': 1,
        }
        self.insert1(key, skip_duplicates=skip_duplicates)


class SplitRFTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.rf_table
        -> self.split_rf_params_table
        ---
        srf: longblob  # spatio receptive field
        trf: longblob  # temporal receptive field
        split_qidx : float  # Quality index as explained variance of the sRF tRF split between 0 and 1
        trf_peak_idxs : blob  # Indexes of peaks in tRF
        '''
        return definition

    rf_table = PlaceholderTable
    split_rf_params_table = PlaceholderTable

    def make(self, key):
        # Get data
        strf = (self.rf_table() & key).fetch1("rf")

        # Get preprocess params
        blur_std, blur_npix, upsample_srf_scale, peak_nstd = \
            self.split_rf_params_table().fetch1('blur_std', 'blur_npix', 'upsample_srf_scale', 'peak_nstd')

        # Get tRF and sRF
        srf, trf = split_strf(strf, blur_std=blur_std, blur_npix=blur_npix, upsample_srf_scale=upsample_srf_scale)

        # Make tRF always positive, so that sRF reflects the polarity of the RF
        polarity, peak_idxs = compute_polarity_and_peak_idxs(trf, nstd=peak_nstd)
        if polarity == -1:
            srf *= -1
            trf *= -1

        strf_fit = merge_strf(srf=resize_srf(srf, output_shape=strf.shape[1:]), trf=trf)
        split_qidx = compute_explained_rf(strf, strf_fit)

        # Save
        rf_key = deepcopy(key)
        rf_key['srf'] = srf
        rf_key['trf'] = trf
        rf_key['trf_peak_idxs'] = peak_idxs
        rf_key['split_qidx'] = split_qidx
        self.insert1(rf_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        dt = (self.rf_table() & key).fetch1("dt")
        srf, trf, peak_idxs = (self & key).fetch1("srf", "trf", "trf_peak_idxs")

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        ax = axs[0]
        plot_srf(srf, ax=ax)

        ax = axs[1]
        plot_trf(trf, dt=dt, peak_idxs=peak_idxs, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        trf = np.stack((self & restriction).fetch('trf'))
        ax = plot_signals_heatmap(signals=trf)
        ax.set(title='tRF')
        plt.show()


class Fit2DRFBaseTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: longblob
        rf_cdia_um: float # Circle equivalent diameter
        rf_area_um2: float # Area covered by 2 standard deviations
        rf_qidx: float # Quality index as explained variance of the sRF estimation between 0 and 1
        """
        return definition

    split_rf_table = PlaceholderTable
    stimulus_table = PlaceholderTable

    def make(self, key):
        raise NotImplementedError()

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        srf = (self.split_rf_table() & key).fetch1("srf")
        srf_fit, rf_qidx = (self & key).fetch1("srf_fit", 'rf_qidx')

        vabsmax = np.maximum(np.max(np.abs(srf)), np.max(np.abs(srf_fit)))

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        ax = axs[0]
        plot_srf(srf, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF')

        ax = axs[1]
        plot_srf(srf_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF fit')

        ax = axs[2]
        plot_srf(srf - srf_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title(f'Difference: QI={rf_qidx:.2f}')

        plt.tight_layout()
        plt.show()

    def plot(self, restriction=None, qi_threshold=0.7):
        if restriction is None:
            restriction = dict()

        rf_qidx, rf_cdia_um = (self & restriction).fetch('rf_qidx', 'rf_cdia_um')

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))

        ax = axs[0]
        ax.set(xlabel="rf_qidx")
        ax.hist(rf_qidx)

        ax = axs[1]
        ax.set(xlabel="rf_cdia_um", title=f'rf_qidx>{qi_threshold}')
        ax.hist(rf_cdia_um[rf_qidx >= qi_threshold])

        plt.tight_layout()
        plt.show()


class FitGauss2DRFTemplate(Fit2DRFBaseTemplate):
    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: longblob
        srf_params: longblob
        rf_cdia_um: float # Circle equivalent diameter
        rf_area_um2: float # Area covered by 2 standard deviations
        rf_qidx: float # Quality index as explained variance of the sRF estimation between 0 and 1
        """
        return definition

    polarity = None

    def make(self, key):
        srf = (self.split_rf_table() & key).fetch1("srf")

        # Get stimulus parameters
        stim_dict = (self.stimulus_table() & key).fetch1("stim_dict")
        pix_scale_x_um = stim_dict["pix_scale_x_um"]
        piy_scale_y_um = stim_dict["pix_scale_y_um"]

        srf_fit, srf_params = fit_rf_model(srf, kind='gauss', polarity=self.polarity)

        x_std = srf_params['x_stddev']
        y_std = srf_params['y_stddev']

        area = np.pi * (2. * x_std * pix_scale_x_um) * (2. * y_std * piy_scale_y_um)
        diameter = np.sqrt(area / np.pi) * 2

        qi = compute_explained_rf(srf, srf_fit)

        # Save
        fit_key = deepcopy(key)
        fit_key['srf_fit'] = srf_fit
        fit_key['srf_params'] = srf_params
        fit_key['rf_cdia_um'] = diameter
        fit_key['rf_area_um2'] = area
        fit_key['rf_qidx'] = qi

        self.insert1(fit_key)


def compute_surround_index(srf, srf_center_fit):
    return np.sum(srf - srf_center_fit) / np.sum(np.abs(srf))


class FitDoG2DRFTemplate(Fit2DRFBaseTemplate):

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: longblob
        srf_center_fit: longblob
        srf_surround_fit: longblob
        srf_params: longblob
        rf_cdia_um: float # Circle equivalent diameter
        rf_area_um2: float # Area covered by 2 standard deviations
        surround_index: float # Strength of surround in sRF
        rf_qidx: float # Quality index as explained variance of the sRF estimation between 0 and 1
        """
        return definition

    polarity = None

    def make(self, key):
        srf = (self.split_rf_table() & key).fetch1("srf")

        # Get stimulus parameters
        stim_dict = (self.stimulus_table() & key).fetch1("stim_dict")
        pix_scale_x_um = stim_dict["pix_scale_x_um"]
        piy_scale_y_um = stim_dict["pix_scale_y_um"]

        srf_fit, srf_center_fit, srf_surround_fit, srf_params = fit_rf_model(
            srf, kind='dog', polarity=self.polarity, bind_mean=True, bind_cov=True)

        x_std = srf_params['x_stddev_0']
        y_std = srf_params['y_stddev_0']

        area = np.pi * (2. * x_std * pix_scale_x_um) * (2. * y_std * piy_scale_y_um)
        diameter = np.sqrt(area / np.pi) * 2

        qi = compute_explained_rf(srf, srf_fit)
        surround_index = compute_surround_index(srf, srf_center_fit)

        # Save
        fit_key = deepcopy(key)
        fit_key['srf_fit'] = srf_fit
        fit_key['srf_center_fit'] = srf_center_fit
        fit_key['srf_surround_fit'] = srf_surround_fit
        fit_key['srf_params'] = srf_params
        fit_key['rf_cdia_um'] = diameter
        fit_key['rf_area_um2'] = area
        fit_key['surround_index'] = surround_index
        fit_key['rf_qidx'] = qi

        self.insert1(fit_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        srf = (self.split_rf_table() & key).fetch1("srf")
        srf_fit, srf_center_fit, srf_surround_fit, rf_qidx = (self & key).fetch1(
            "srf_fit", 'srf_center_fit', 'srf_surround_fit', 'rf_qidx')

        vabsmax = np.maximum(np.max(np.abs(srf)), np.max(np.abs(srf_fit)))

        fig, axs = plt.subplots(1, 5, figsize=(20, 3))
        ax = axs[0]
        plot_srf(srf, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF')

        ax = axs[1]
        plot_srf(srf_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF fit')

        ax = axs[2]
        plot_srf(srf - srf_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title(f'Difference: QI={rf_qidx:.2f}')

        ax = axs[3]
        plot_srf(srf_center_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF center fit')

        ax = axs[4]
        plot_srf(srf_surround_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF surround fit')

        plt.tight_layout()
        plt.show()


def fit_rf_model(srf, kind='gaussian', polarity=None, **model_kws):
    assert srf.ndim == 2, 'Provide 2d RF'

    if kind == 'gauss':
        model = srf_gauss_model(srf, polarity=polarity)
    elif kind == 'dog':
        model = srf_dog_model(srf, polarity=polarity, **model_kws)
    else:
        raise NotImplementedError(kind)

    xx, yy = np.meshgrid(np.arange(0, srf.shape[1]), np.arange(0, srf.shape[0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SLSQPLSQFitter()(model, xx, yy, srf, verblevel=0)

    model_params = {k: v for k, v in zip(model.param_names, model.param_sets.flatten())}
    model_fit = model(xx, yy)

    if kind == 'gauss':
        return model_fit, model_params
    elif kind == 'dog':
        model_center_fit = model[0](xx, yy)
        model_surround_fit = model[1](xx, yy)
        return model_fit, model_center_fit, model_surround_fit, model_params


def estimate_srf_center_model_init_params(srf, polarity=None, amplimscale=2.):
    assert srf.ndim == 2, 'Provide 2d RF'

    if polarity is None:
        y0, x0 = np.unravel_index(np.argmax(np.abs(srf)), shape=srf.shape)
    elif polarity == 1:
        y0, x0 = np.unravel_index(np.argmax(srf), shape=srf.shape)
    elif polarity == -1:
        y0, x0 = np.unravel_index(np.argmin(srf), shape=srf.shape)
    else:
        raise NotImplementedError(polarity)

    srf_cut = np.abs(srf[:, int(x0)] - np.min(srf[:, int(x0)]))
    srf_cut_profile = np.cumsum(srf_cut) / np.sum(srf_cut)
    std0 = np.maximum(1, 0.5 * (np.argmin(np.abs(srf_cut_profile - 0.7)) - np.argmin(np.abs(srf_cut_profile - 0.3))))

    xlim = (0, srf.shape[1])
    ylim = (0, srf.shape[0])

    if polarity is None:
        amp0 = srf.flat[np.argmax(np.abs(srf))]
        amplim = (-amplimscale * np.abs(amp0), amplimscale * np.abs(amp0))
    elif polarity == 1:
        amp0 = np.max(srf)
        assert amp0 > 0
        amplim = (0., amplimscale * amp0)
    elif polarity == -1:
        amp0 = np.min(srf)
        assert amp0 < 0
        amplim = (amplimscale * amp0, 0.)
    else:
        raise NotImplementedError(polarity)

    return x0, y0, amp0, std0, xlim, ylim, amplim


def srf_gauss_model(srf, polarity=None):
    assert srf.ndim == 2, 'Provide 2d RF'

    x0, y0, amp0, std0, xlim, ylim, amplim = estimate_srf_center_model_init_params(
        srf=srf, polarity=polarity, amplimscale=2.)

    model = Gaussian2D(
        amplitude=amp0, x_mean=x0, y_mean=y0, x_stddev=std0, y_stddev=std0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim})

    return model


def srf_dog_model(srf, polarity=None, bind_mean=True, bind_cov=False):
    assert srf.ndim == 2, 'Provide 2d RF'

    x0, y0, amp0, std0, xlim, ylim, amplim = estimate_srf_center_model_init_params(
        srf=srf, polarity=polarity, amplimscale=10.)

    xlim = (np.maximum(xlim[0], x0 - std0), np.minimum(xlim[1], x0 + std0))
    ylim = (np.maximum(ylim[0], y0 - std0), np.minimum(ylim[1], y0 + std0))
    stdlim = (0.1 * std0, 10. * std0)

    f_c = Gaussian2D(
        amplitude=amp0, x_mean=x0, y_mean=y0, x_stddev=std0, y_stddev=std0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim, 'x_stddev': stdlim, 'y_stddev': stdlim})

    f_s = Gaussian2D(
        amplitude=1. / 50. * amp0, x_mean=x0, y_mean=y0, x_stddev=5. * std0, y_stddev=5. * std0,
        bounds={'amplitude': np.sort(-np.asarray(amplim)), 'x_mean': xlim, 'y_mean': ylim})

    model = f_c - f_s

    if bind_mean:
        f_s.x_mean.tied = lambda _model: _model.x_mean_0
        f_s.y_mean.tied = lambda _model: _model.y_mean_0
    if bind_cov:
        f_s.theta.tied = lambda _model: _model.theta_0
        f_s.x_stddev.tied = lambda _model: _model.y_stddev_1 * _model.x_stddev_0 / _model.y_stddev_0

    return model
