import os
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import PlaceholderTable
from djimaging.utils.data_utils import list_data_files
from djimaging.utils.scanm_utils import load_ch0_ch1_stacks_from_h5, load_ch0_ch1_stacks_from_smp, get_pixel_size_um


def load_high_res_stack(pre_data_path, raw_data_path, field, field_loc, highres_alias):
    """Scan filesystem for file and load data. Tries to load h5 files first, but may also load raw files."""
    filepath_h5 = scan_for_highres_filepath(
        folder=pre_data_path, field=field, field_loc=field_loc, highres_alias=highres_alias, ftype='h5')

    filepath_smp = scan_for_highres_filepath(
        folder=raw_data_path, field=field, field_loc=field_loc-1, highres_alias=highres_alias, ftype='smp')

    if filepath_h5 is not None:
        filepath = filepath_h5
        ch0_stack, ch1_stack, wparams = load_ch0_ch1_stacks_from_h5(filepath_h5)
    elif filepath_smp is not None:
        filepath = filepath_smp
        ch0_stack, ch1_stack, wparams = load_ch0_ch1_stacks_from_smp(filepath_smp)
    else:
        filepath, ch0_stack, ch1_stack, wparams = None, None, None, None

    return filepath, ch0_stack, ch1_stack, wparams


def scan_for_highres_filepath(folder, field, field_loc, highres_alias, ftype='h5'):
    """Scan filesystem for files that match the highres alias and are from the same field."""
    data_files = list_data_files(folder=folder, hidden=False, field=field, field_loc=field_loc, ftype=ftype)
    for filename in data_files:
        for alias in highres_alias.split('_'):
            for fileinfo in filename.replace(f'.{ftype}', '').split('_')[field_loc:]:
                if alias.lower() == fileinfo.lower():
                    filepath = os.path.join(folder, filename)
                    return filepath
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
            field=field, field_loc=field_loc, highres_alias=highres_alias)

        if filepath is None or ch0_stack is None or ch1_stack is None:
            return

        # Get pixel sizes
        nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
        nypix = wparams["user_dypix"]
        pixel_size_um = get_pixel_size_um(zoom=wparams["zoom"], setupid=setupid, nypix=nypix)

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
        highres_key["nxpix_offset"] = wparams["user_nxpixlineoffs"]
        highres_key["nxpix_retrace"] = wparams["user_npixretrace"]

        highres_key["zoom"] = wparams["zoom"]
        highres_key["pixel_size_um"] = pixel_size_um

        highres_key["nframes"] = ch0_stack.shape[2]

        self.insert1(highres_key)

    def plot1(self, key: dict):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        ch0_average = (self & key).fetch1("ch0_average").T
        ch1_average = (self & key).fetch1("ch1_average").T

        axs[0].imshow(ch0_average)
        axs[0].set(title='ch0_average')

        axs[1].imshow(ch1_average)
        axs[1].set(title='ch1_average')

        plt.show()
