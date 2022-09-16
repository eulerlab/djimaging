import os
import random
import warnings
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import PlaceholderTable
from djimaging.utils.alias_utils import match_file, get_field_files
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm_utils import load_ch0_ch1_stacks_from_h5, load_ch0_ch1_stacks_from_smp, get_pixel_size_xy_um


def load_high_res_stack(pre_data_path, raw_data_path, highres_alias,
                        field, field_loc, condition=None, condition_loc=None):
    """Scan filesystem for file and load data. Tries to load h5 files first, but may also load raw files."""

    filepath = scan_for_highres_filepath(
        folder=pre_data_path, highres_alias=highres_alias, field=field, field_loc=field_loc,
        condition=condition, condition_loc=condition_loc, ftype='h5')

    if filepath is not None:
        try:
            ch0_stack, ch1_stack, wparams = load_ch0_ch1_stacks_from_h5(filepath)
            return filepath, ch0_stack, ch1_stack, wparams
        except OSError:
            warnings.warn(f'OSError when reading file: {filepath}')
            pass

    filepath = scan_for_highres_filepath(
        folder=raw_data_path, highres_alias=highres_alias, field=field, field_loc=field_loc,
        condition=condition, condition_loc=condition_loc, ftype='smp')

    if filepath is not None:
        try:
            ch0_stack, ch1_stack, wparams = load_ch0_ch1_stacks_from_smp(filepath)
            return filepath, ch0_stack, ch1_stack, wparams
        except OSError:
            warnings.warn(f'OSError when reading file: {filepath}')
            pass

    return None, None, None, None


def scan_for_highres_filepath(folder, field, field_loc, highres_alias, condition=None, condition_loc=None, ftype='h5'):
    """Scan filesystem for files that match the highres alias and are from the same field."""
    if not os.path.isdir(folder):
        return None

    field_files = get_field_files(folder=folder, field=field, field_loc=field_loc, incl_hidden=False, ftype=ftype)

    for file in field_files:
        is_highres = match_file(file, pattern=highres_alias, pattern_loc=None, ftype=ftype)
        is_condition = True if condition is None else \
            match_file(file, pattern=condition, pattern_loc=condition_loc, ftype=ftype)

        if is_highres & is_condition:
            return os.path.join(folder, file)
    return None


class HighResTemplate(dj.Computed):

    @property
    def definition(self):
        definition = """
        # High resolution stack information.
        -> self.field_table
        ---
        fromfile : varchar(255)  # Absolute path to file
        ch0_average : longblob  # Stack median of the high resolution image channel 0
        ch1_average : longblob  # Stack median of the high resolution image channel 1
        nframes : int  # Number of frames averaged
        absx: float  # absolute position of the center (of the cropped field) in the x axis as recorded by ScanM
        absy: float  # absolute position of the center (of the cropped field) in the y axis as recorded by ScanM
        absz: float  # absolute position of the center (of the cropped field) in the z axis as recorded by ScanM
        nxpix: int  # number of pixels in x
        nypix: int  # number of pixels in y
        nzpix: int  # number of pixels in z
        nxpix_offset: int  # number of offset pixels in x
        nxpix_retrace: int  # number of retrace pixels in x
        zoom: float  # zoom factor used during recording
        pixel_size_um :float  # width / height of a pixel in um        
        """
        return definition

    field_table = PlaceholderTable
    experiment_table = PlaceholderTable
    userinfo_table = PlaceholderTable

    def make(self, key):
        field = (self.field_table() & key).fetch1("field")
        stim_loc, field_loc, condition_loc = (self.userinfo_table() & key).fetch1(
            "stimulus_loc", "field_loc", "condition_loc")
        highres_alias = (self.userinfo_table() & key).fetch1("highres_alias")
        header_path = (self.experiment_table() & key).fetch1('header_path')
        pre_data_path = os.path.join(header_path, (self.userinfo_table() & key).fetch1("pre_data_dir"))
        raw_data_path = os.path.join(header_path, (self.userinfo_table() & key).fetch1("raw_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"
        setupid = (self.experiment_table.ExpInfo() & key).fetch1("setupid")

        filepath, ch0_stack, ch1_stack, wparams = load_high_res_stack(
            pre_data_path=pre_data_path, raw_data_path=raw_data_path,
            highres_alias=highres_alias, field=field, field_loc=field_loc)

        if filepath is None or ch0_stack is None or ch1_stack is None:
            return

        # Get pixel sizes
        nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
        nypix = wparams["user_dypix"]
        nzpix = wparams["user_dzpix"]
        pixel_size_um = get_pixel_size_xy_um(zoom=wparams["zoom"], setupid=setupid, npix=nxpix)

        # Insert key
        highres_key = deepcopy(key)
        highres_key["fromfile"] = filepath
        highres_key["ch0_average"] = np.median(ch0_stack, 2)
        highres_key["ch1_average"] = np.median(ch1_stack, 2)

        highres_key["absx"] = wparams['xcoord_um']
        highres_key["absy"] = wparams['ycoord_um']
        highres_key["absz"] = wparams['zcoord_um']

        highres_key["nxpix"] = nxpix
        highres_key["nypix"] = nypix
        highres_key["nzpix"] = nzpix
        highres_key["nxpix_offset"] = wparams["user_nxpixlineoffs"]
        highres_key["nxpix_retrace"] = wparams["user_npixretrace"]

        highres_key["zoom"] = wparams["zoom"]
        highres_key["pixel_size_um"] = pixel_size_um

        highres_key["nframes"] = ch0_stack.shape[2]

        self.insert1(highres_key)

    def plot1(self, key=None, figsize=(8, 4)):
        if key is not None:
            key = {k: v for k, v in key.items() if k in self.primary_key}
        else:
            key = random.choice(self.fetch(*self.primary_key, as_dict=True))

        ch0_average = (self & key).fetch1("ch0_average").T
        ch1_average = (self & key).fetch1("ch1_average").T

        plot_field(ch0_average, ch1_average, roi_mask=None, title=key, figsize=figsize)
