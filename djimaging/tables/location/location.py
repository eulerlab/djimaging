"""
Tables for location of the recorded fields relative to the optic disk.

Example usage:

from djimaging.tables import location

@schema
class OpticDisk(location.OpticDiskTemplate):
    incl_region = True # Include region as primary key? If so you have to provide the field_table:
    field_table = Field

    userinfo_table = UserInfo
    experiment_table = Experiment
    raw_params_table = RawDataParams


@schema
class RelativeFieldLocation(location.RelativeFieldLocationTemplate):
    field_table = Field
    presentation_table = Presentation
    opticdisk_table = OpticDisk


@schema
class RetinalFieldLocation(location.RetinalFieldLocationTemplate):
    relativefieldlocation_table = RelativeFieldLocation
    retina_orientation_table = RetinaOrientation
"""

import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.scanm.setup_utils import get_retinal_position


class RelativeFieldLocationTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of the recorded field center relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer to curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right
        
        -> self.opticdisk_table
        -> self.field_table
        ---
        relx   :float32      # XCoord_um relative to the optic disk
        rely   :float32      # YCoord_um relative to the optic disk
        relz   :float32      # ZCoord_um relative to the optic disk
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.opticdisk_table.proj() * self.field_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def opticdisk_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    def presentation_table(self):  # Optional table
        return None

    def make(self, key: dict) -> None:
        """Compute the field position relative to the optic disk and insert it.

        Args:
            key: DataJoint primary key dict identifying a unique combination of optic
                disk and field entries.
        """
        od_key = key.copy()
        od_key.pop('field', None)

        odx, ody, odz = (self.opticdisk_table() & od_key).fetch1("odx", "ody", "odz")
        absx, absy, absz = (self.field_table() & key).fetch1('absx', 'absy', 'absz')

        if self.presentation_table is not None:
            pres_absx, pres_absy, pres_absz = (self.presentation_table() & key).to_arrays('absx', 'absy', 'absz')
            if np.abs(np.median(pres_absx) - absx) > 200 or np.abs(np.median(pres_absy) - absy) > 200:
                warnings.warn(f"Position is different from the presentation table for {key}. "
                              f"Using heuristic to determine location.")
                outlier_x = np.abs(pres_absx) > 20_000
                outlier_y = np.abs(pres_absy) > 20_000
                outlier_xy = np.logical_or(outlier_x, outlier_y)

                if np.all(outlier_xy):
                    # All values are extremely high, use median and hope for the best
                    absx = np.median(pres_absx)
                    absy = np.median(pres_absy)
                    absz = np.median(pres_absz)
                else:
                    # Use the median of the non-outliers
                    absx = np.median(pres_absx[~outlier_xy])
                    absy = np.median(pres_absy[~outlier_xy])
                    absz = np.median(pres_absz[~outlier_xy])

        loc_key = key.copy()
        loc_key["relx"] = absx - odx
        loc_key["rely"] = absy - ody
        loc_key["relz"] = absz - odz

        self.insert1(loc_key)

    def plot(self, view: str = 'igor_local') -> None:
        """Scatter-plot all field positions in setup coordinates.

        Args:
            view: Axis orientation; one of ``'igor_local'`` or ``'igor_setup'``.
        """
        relx, rely = self.to_arrays("relx", "rely")
        plot_relxy_pos(relx, rely, view=view)


class RetinalFieldLocationTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of the recorded fields relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right

        -> self.relativefieldlocation_table
        -> self.retina_orientation_table
        ---
        ventral_dorsal_pos_um = NULL :float32      # position on the ventral-dorsal axis, greater 0 means dorsal; NULL if unknown
        temporal_nasal_pos_um = NULL :float32      # position on the temporal-nasal axis, greater 0 means nasal; NULL if unknown
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.relativefieldlocation_table.proj() * self.retina_orientation_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def relativefieldlocation_table(self):
        pass

    @property
    @abstractmethod
    def retina_orientation_table(self):
        pass

    def make(self, key: dict) -> None:
        """Convert relative field coordinates to retinal (ventral-dorsal / temporal-nasal) axes.

        Args:
            key: DataJoint primary key dict identifying a unique relative field location entry.
        """
        relx, rely = (self.relativefieldlocation_table() & key).fetch1('relx', 'rely')
        eye, orientation_deg = self.retina_orientation_table().get_eye_and_orientation(key)
        ventral_dorsal_pos_um, temporal_nasal_pos_um = retinal_position_or_none(relx, rely, orientation_deg, eye)

        rfl_key = key.copy()
        rfl_key['ventral_dorsal_pos_um'] = ventral_dorsal_pos_um
        rfl_key['temporal_nasal_pos_um'] = temporal_nasal_pos_um
        self.insert1(rfl_key)

    def plot(self, key: dict = None) -> None:
        """Scatter-plot all fields in retinal coordinates, optionally highlighting one key.

        Args:
            key: Optional DataJoint primary key dict. If provided, the matching entry is
                highlighted with a separate scatter marker.
        """
        temporal_nasal_pos_um, ventral_dorsal_pos_um = self.to_arrays("temporal_nasal_pos_um", "ventral_dorsal_pos_um")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(temporal_nasal_pos_um, ventral_dorsal_pos_um, label='all')
        if key is not None:
            ktemporal_nasal_pos_um, kventral_dorsal_pos_um = (self & key).fetch1(
                "temporal_nasal_pos_um", "ventral_dorsal_pos_um")
            ax.scatter(ktemporal_nasal_pos_um, kventral_dorsal_pos_um, label='key')
            ax.legend()
        ax.set(xlabel="temporal_nasal_pos_um", ylabel="ventral_dorsal_pos_um")
        ax.set_aspect(aspect="equal", adjustable="datalim")
        plt.show()


def retinal_position_or_none(relx: float, rely: float, orientation_deg: float | None, eye: str | None)         -> tuple[float | None, float | None]:
    """(ventral_dorsal_pos_um, temporal_nasal_pos_um) from positions relative to the optic disk, None where unknown.

    Both are None if the orientation is unknown, temporal_nasal_pos_um is None if the eye is unknown.
    NaN cannot be inserted by DataJoint (the column is omitted), so unknown values are None (NULL).
    """
    if orientation_deg is None:
        return None, None
    ventral_dorsal, temporal_nasal = get_retinal_position(
        rel_xcoord_um=relx, rel_ycoord_um=rely, rotation=orientation_deg, eye=eye)
    return (None if np.isnan(ventral_dorsal) else float(ventral_dorsal),
            None if np.isnan(temporal_nasal) else float(temporal_nasal))


def plot_relxy_pos(relx: np.ndarray, rely: np.ndarray, view: str = 'igor_local') -> None:
    """Plot field positions in setup XY coordinates.

    Args:
        relx: Array of X coordinates relative to the optic disk (in µm).
        rely: Array of Y coordinates relative to the optic disk (in µm).
        view: Axis orientation; one of ``'igor_local'`` (invert both axes) or
            ``'igor_setup'`` (invert y-axis only).

    Raises:
        NotImplementedError: If *view* is not recognised.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(rely, relx, label='all', s=1, alpha=0.5)
    ax.set(xlabel="rely", ylabel="relx")

    if view == 'igor_local':
        ax.invert_xaxis()
        ax.invert_yaxis()
    elif view == 'igor_setup':
        ax.invert_yaxis()
    else:
        raise NotImplementedError(view)

    ax.set_aspect(aspect="equal", adjustable="datalim")

    plt.show()


def plot_relyz_pos(rely: np.ndarray, relz: np.ndarray, view: str = 'igor_local') -> None:
    """Plot field positions in setup YZ coordinates.

    Args:
        rely: Array of Y coordinates relative to the optic disk (in µm).
        relz: Array of Z coordinates relative to the optic disk (in µm).
        view: Axis orientation; ``'igor_local'`` inverts the x-axis, ``'igor_setup'``
            leaves axes as-is.

    Raises:
        NotImplementedError: If *view* is not recognised.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(rely, relz, label='all', s=1, alpha=0.5)
    ax.set(xlabel="rely", ylabel="relz")

    if view == 'igor_local':
        ax.invert_xaxis()
    elif view == 'igor_setup':
        pass
    else:
        raise NotImplementedError(view)

    ax.set_aspect(aspect="equal", adjustable="datalim")
    plt.show()
