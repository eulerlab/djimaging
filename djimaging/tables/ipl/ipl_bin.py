"""
Example usage:

from djimaging.tables import ipl

@schema
class IplBinParams(ipl.IplBinParamsTemplate):
    pass


@schema
class RoiIplBin(ipl.RoiIplBinTemplate):
    roi_ipl_table = RoiIplDepth
    ipl_bin_params_table = IplBinParams
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np


class IplBinParamsTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
            ipl_bin_id : tinyint unsigned # ipl bin id
            ---
            n_bins : tinyint unsigned # number of bins
            bin_borders   :blob    # pixel index where gcl/ipl border intersects on left of image (with GCL up)
            bin_names  = NULL :blob    # pixel index where gcl/ipl border intersects on right side of image
            """
        return definition

    def add(self, ipl_bin_id, n_bins, bin_borders, bin_names=None, skip_duplicates=False):

        if len(bin_borders) != n_bins + 1:
            raise ValueError(f"len(bin_borders)={len(bin_borders)} must be n_bins+1={n_bins + 1}")
        if bin_names is not None and len(bin_names) != n_bins:
            raise ValueError(f"len(bin_names)={len(bin_names)} must be n_bins={n_bins}")

        if not isinstance(bin_borders, np.ndarray):
            bin_borders = np.asarray(bin_borders)

        if not np.all(np.sort(bin_borders) == bin_borders):
            raise ValueError("bin_borders must be sorted")

        self.insert1(dict(ipl_bin_id=ipl_bin_id, n_bins=n_bins, bin_borders=bin_borders, bin_names=bin_names),
                     skip_duplicates=True)


class RoiIplBinTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
            -> self.roi_ipl_table
            -> self.ipl_bin_params_table
            ---
            roi_ipl_bin : tinyint unsigned # ipl bin id -1 if too low, n_bins if too high
            """
        return definition

    @property
    @abstractmethod
    def roi_ipl_table(self):
        pass

    @property
    @abstractmethod
    def ipl_bin_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.roi_ipl_table.proj() * self.ipl_bin_params_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        roi_ipl_depth = (self.roi_ipl_table & key).fetch1('ipl_depth')
        bin_borders = (self.ipl_bin_params_table & key).fetch1('bin_borders')
        roi_ipl_bin = np.digitize(roi_ipl_depth, bin_borders) - 1
        self.insert1(dict(**key, roi_ipl_bin=roi_ipl_bin))
