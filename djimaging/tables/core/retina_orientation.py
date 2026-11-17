"""
Eye and orientation of the retina in the recording chamber.

This table is the single source of eye and orientation for everything that maps chamber (XCoord, YCoord) positions or
directions onto the retina (retinal field/ROI locations, outlines, moving bar directions). Unknown values are NULL.

Values are read from the experiment header (`Experiment.ExpInfo`). Use `set_manual` to correct them.

Example usage:

from djimaging.tables import core

@schema
class RetinaOrientation(core.RetinaOrientationTemplate):
    incl_region = False  # One entry per region (e.g. LR, RR) instead of per experiment? Then bind field_table.
    experiment_table = Experiment
    field_table = Field  # Only used if incl_region
"""
import os
import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.scanm.read_h5_utils import read_config_dict


class RetinaOrientationTemplate(dj.Computed):
    database = ""
    incl_region = False  # One entry per region of the Field table instead of one per experiment
    _orientation_if_missing = 0.  # Orientation used if the header has none (source 'header_default'); None: NULL

    @property
    def definition(self):
        definition = """
        # eye and orientation of the retina in the recording chamber; NULL if unknown
        -> self.experiment_table
        """
        if self.incl_region:
            definition += "    region   :varchar(16)    # region (e.g. LR or RR)\n"
        definition += """
        ---
        eye = NULL               : enum('left', 'right')  # eye of the retina; NULL if unknown
        orientation_deg = NULL   : float32  # in the chamber, 0 = dorsal away from experimenter; NULL if unknown
        source                   : enum('header', 'header_default', 'manual')  # header_default: not in header
        """
        return definition

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    def field_table(self):  # Optional, only used if incl_region
        return None

    @property
    def key_source(self):
        try:
            if self.incl_region:
                return dj.U(*self.experiment_table.primary_key, 'region') & self.field_table
            return self.experiment_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        eye, prepwmorient = (self.experiment_table.ExpInfo & key).fetch1('eye', 'prepwmorient')
        header_file = os.path.join(*(self.experiment_table & key).fetch1('header_path', 'header_name'))
        entry = dict(**key, **orientation_from_header(
            eye=eye, prepwmorient=prepwmorient, header_has_orientation=header_has_orientation(header_file),
            orientation_if_missing=self._orientation_if_missing))
        if entry['source'] == 'header_default':
            warnings.warn(f"No retina orientation in header of {key}. Using {self._orientation_if_missing}. "
                          f"Correct with RetinaOrientation().set_manual(...) if this is wrong.")
        self.insert1(entry)

    def set_manual(self, key: dict, eye: str | None, orientation_deg: float | None) -> None:
        """Set eye and orientation of an experiment (or region) manually, e.g. if the header is wrong.

        Deletes the existing entry and all downstream entries (retinal locations, moving bar directions, ...),
        which have to be populated again. Use None for unknown.
        """
        if eye not in ('left', 'right', None):
            raise ValueError(f"eye must be 'left', 'right' or None, not {eye!r}")
        keys = (self.key_source & key).proj().to_dicts()
        if len(keys) != 1:
            raise ValueError(f"key must identify exactly one {self.__class__.__name__} entry, found {len(keys)}")
        (self & keys[0]).delete()
        self.insert1(dict(**keys[0], eye=eye, orientation_deg=_none_if_nan(orientation_deg), source='manual'),
                     allow_direct_insert=True)

    def get_eye_and_orientation(self, key: dict) -> tuple[str | None, float | None]:
        """Eye ('left', 'right' or None) and orientation in degrees (or None) for a key of any downstream table."""
        rows = (self & key).to_dicts()
        if len(rows) != 1:
            raise ValueError(f"Found {len(rows)} {self.__class__.__name__} entries for {key}, expected 1. "
                             f"Populate it, and if incl_region is set, restrict by region.")
        orientation_deg = rows[0]['orientation_deg']
        return rows[0]['eye'], (None if orientation_deg is None else float(orientation_deg))


def header_has_orientation(header_file: str) -> bool | None:
    """Whether the header defines the orientation (`prepwmorient`); None if the header cannot be read."""
    try:
        if not os.path.isfile(header_file):  # ConfigParser silently ignores missing files
            raise FileNotFoundError(header_file)
        return read_config_dict(header_file).get('prepwmorient', '') != ''
    except (OSError, ValueError, KeyError) as e:
        warnings.warn(f"Could not read header {header_file}: {e}")
        return None


def orientation_from_header(eye: str, prepwmorient, header_has_orientation: bool | None,
                            orientation_if_missing: float | None = 0.) -> dict:
    """Eye, orientation and source from the `Experiment.ExpInfo` values.

    `ExpInfo.prepwmorient` is -1 for unknown and 0 if not given in the header; `header_has_orientation` tells the two
    0 cases apart (None: header not readable, trust ExpInfo).
    """
    eye = eye.lower() if isinstance(eye, str) and eye.lower() in ('left', 'right') else None
    if header_has_orientation is False:
        return dict(eye=eye, orientation_deg=_none_if_nan(orientation_if_missing), source='header_default')
    orientation_deg = None if prepwmorient is None or int(prepwmorient) == -1 else float(prepwmorient)
    return dict(eye=eye, orientation_deg=orientation_deg, source='header')


def _none_if_nan(value):
    return None if value is None or not np.isfinite(value) else float(value)
