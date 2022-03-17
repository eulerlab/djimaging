from djimaging.utils.data_utils import load_h5_data
import numpy as np
import os


def add_dentrendparams_defaults(dentrendparams_table):
    """Add default detrend parameter to table"""
    detrendparams_key = {
        'detrend_param_set_id': 1,
        'window_length': 60,
        'poly_order': 3,
        'non_negative': 0,
        'subtract_baseline': 1,
        'standardize': 1,
    }

    if detrendparams_key['detrend_param_set_id'] not in dentrendparams_table().fetch('detrend_param_set_id'):
        dentrendparams_table().insert1(detrendparams_key)


def add_stimulus_defaults(stimulus_table):
    """Add default stimuli to stimulus table"""
    add_no_stim(stimulus_table)
    add_movingbar(stimulus_table)
    add_chirp(stimulus_table)
    add_noise(stimulus_table)


def add_no_stim(stimulus_table):
    # no_stim
    key = {
        "stim_id": -1,
        "stim_v": 0,
        "stimulusname": 'no_stim',
        "stim_path": '',
        "is_colour": 0,
        "framerate": 0,
        "commit_id": "",
        "alias": 'nostim_none',
        "ntrigger_rep": 0,
        "isrepeated": 0,
    }

    if key['stim_id'] not in stimulus_table().fetch('stim_id'):
        stimulus_table().insert1(key)


def add_noise(stimulus_table, stim_path='/gpfs01/euler/data/Resources/Stimulus/old/noise.h5'):
    assert os.path.isfile(stim_path), stim_path
    stimulus_trace = load_h5_data(stim_path)['k']

    stimulus_info = {
        "pix_n_x": 20,
        "pix_n_y": 15,
        "pix_scale_x_um": 30,
        "pix_scale_y_um": 30,
    }

    key = {
        "stim_id": 0,
        "stim_v": 0,
        "stimulusname": 'noise',
        "stim_path": stim_path,
        "is_colour": 0,
        "stimulus_trace": stimulus_trace,
        "stimulus_info": str(stimulus_info),
        "framerate": 5,
        "commit_id": "",
        "alias": "noise_dn_noise10m_noise20m_noise30m_noise40m",
        "ntrigger_rep": 1500,
        "isrepeated": 0,
    }

    if key['stim_id'] not in stimulus_table().fetch('stim_id'):
        stimulus_table().insert1(key)


def add_chirp(stimulus_table, stim_path='/gpfs01/euler/data/Resources/Stimulus/old/chirp_old.h5'):
    assert os.path.isfile(stim_path)
    stimulus_trace = load_h5_data(stim_path)['chirp']

    stimulus_info = {
        "diameter_um": 1000,
    }

    key = {
        "stim_id": 1,
        "stim_v": 0,
        "stimulusname": 'chirp',
        "stim_path": stim_path,
        "is_colour": 0,
        "stimulus_trace": stimulus_trace,
        "stimulus_info": str(stimulus_info),
        "framerate": 1 / 60.,
        "commit_id": "",
        "alias": "chirp_gchirp_globalchirp",
        "ntrigger_rep": 2,
        "isrepeated": 1,
    }

    if key['stim_id'] not in stimulus_table().fetch('stim_id'):
        stimulus_table().insert1(key)


def add_movingbar(stimulus_table, stim_path=''):
    key = {
        "stim_id": 2,
        "stim_v": 0,
        "stimulusname": 'movingbar',
        "stim_path": stim_path,
        "is_colour": 0,
        "trial_info": np.array([0, 180, 45, 225, 90, 270, 135, 315]),
        "stimulus_info": "",
        "framerate": 1 / 60.,
        "commit_id": "",
        "alias": "mb_mbar_bar_movingbar",
        "ntrigger_rep": 1,
        "isrepeated": 1,
    }

    if key['stim_id'] not in stimulus_table().fetch('stim_id'):
        stimulus_table().insert1(key)
