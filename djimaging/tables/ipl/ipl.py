from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.scanm_utils import get_roi_centers


class IplBordersTemplate(dj.Manual):
    @property
    def definition(self):
        definition = """
            # Manually determined index and thickness information for the IPL.
            USe XZ widget notebook to determine values.
            -> field_or_pres_table
            ---
            left   :tinyint    # pixel index where gcl/ipl border intersects on left of image (with GCL up)
            right  :tinyint    # pixel index where gcl/ipl border intersects on right side of image
            thick  :tinyint    # pixel width of the ipl
            """
        return definition

    @property
    @abstractmethod
    def field_or_pres_table(self):
        pass


class RoiIplDepthTemplate(dj.Computed):
    @property
    def definition(self):
        definition = """
        # Gives the IPL depth of the field's ROIs relative to the GCL (=0) and INL (=1)
        -> ipl_border_table
        -> roi_table
        ---
        ipl_depth : float  # Depth in the IPL relative to the GCL (=0) and INL (=1)
        """
        return definition

    @property
    @abstractmethod
    def ipl_border_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    def make(self, key):
        roi_ids = (self.roi_table & key).fetch("roi_id")
        left, right, thick = (self.ipl_border_table & key).fetch1('left', 'right', 'thick')
        roi_mask = (self.ipl_border_table.field_or_pres_table.RoiMask & key).fetch1('roi_mask')

        roi_centers = get_roi_centers(roi_mask, roi_ids)

        # calculate the depth relative to the IPL borders
        m1, b1 = self.get_line([(0, left), (roi_mask.shape[0] - 1, right)])

        for roi_id in roi_ids:
            shifts = m1 * roi_centers[:, 0] + b1
            ipl_depth = (roi_centers[:, 1] - shifts) / thick

            self.insert1(dict(**key, roi_id=roi_id, ipl_depth=ipl_depth))

    @staticmethod
    def get_line(points):
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        return m, c