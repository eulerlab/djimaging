"""
Setup tables, per setup and period, assigned to each experiment from its setupid and date:

- Stimulator: how the QDSpy screen is projected onto the retina (geometry) and the stimulator delay. All stimulus
  geometry (moving bar directions, dense noise RF positions, mouse cam movie orientation) is taken from it, see
  djimaging.utils.stimulator for the frames and conventions.
- ImagingSetup: scan width at zoom 1 (pixel size) and light artifact rows of the microscope.

Field, Presentation, HighRes and OutlineAbs read the stimulator delay, scan width and light artifact rows from these
tables if `experiment_stimulator_table` and `experiment_imaging_setup_table` are bound (see
djimaging.tables.core.setup_params), so the setup tables must be defined and populated before Field.

Example usage:

@schema
class Stimulator(stimulator.StimulatorTemplate):
    pass


@schema
class ExperimentStimulator(stimulator.ExperimentStimulatorTemplate):
    experiment_table = Experiment
    stimulator_table = Stimulator


@schema
class ImagingSetup(stimulator.ImagingSetupTemplate):
    pass


@schema
class ExperimentImagingSetup(stimulator.ExperimentImagingSetupTemplate):
    experiment_table = Experiment
    imaging_setup_table = ImagingSetup

Stimulator().add_defaults()
ImagingSetup().add_defaults()
ExperimentStimulator().populate()
ExperimentImagingSetup().populate()
ExperimentStimulator().report_unassigned()
"""
import warnings
from abc import abstractmethod

import datajoint as dj

from djimaging.utils.stimulator import (
    DEFAULT_IMAGING_SETUPS, DEFAULT_STIMULATORS, assign_imaging_setup, assign_stimulator, screen_to_field_matrix)


class StimulatorTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        # Stimulator of one setup and period: projection of the QDSpy screen onto the retina and delay
        stimulator_id          : varchar(32)    # e.g. 'setup1_2019-12-12'
        ---
        setupid                : varchar(191)   # setupid as in Experiment.ExpInfo
        projection             : varchar(255)   # description of the stimulator and projection path
        valid_from             : date           # first experiment date this stimulator applies to
        valid_to = null        : date           # last experiment date (NULL: open end)
        documented_from = null : date           # documented period of the geometry; outside: medium confidence
        documented_to = null   : date           # NULL for both: geometry not documented
        screen_x_axis          : enum('+x', '-x', '+y', '-y')  # field axis (x=XCoord, y=YCoord) of QDSpy screen +x
        screen_y_axis          : enum('+x', '-x', '+y', '-y')  # field axis (x=XCoord, y=YCoord) of QDSpy screen +y
        offset_relx_um = 0     : float32        # stimulus center minus scan field center along XCoord_um
        offset_rely_um = 0     : float32        # stimulus center minus scan field center along YCoord_um
        um_per_px = null       : float32        # size of one stimulator pixel on the retina (QDSpy config)
        stimulator_delay_ms    : float32        # delay between trigger and stimulus on the retina
        color_map              : varchar(63)    # mapping of the RGB input to the LEDs
        confidence             : enum('high', 'medium', 'low')  # confidence in the geometry
        source                 : varchar(511)   # where geometry and delay are documented
        """
        return definition

    def add(self, stimulator_id: str, setupid: str, screen_x_axis: str, screen_y_axis: str, confidence: str,
            source: str, stimulator_delay_ms: float, projection: str = '', valid_from: str = '2000-01-01',
            valid_to: str | None = None, documented_from: str | None = None, documented_to: str | None = None,
            offset_relx_um: float = 0., offset_rely_um: float = 0., um_per_px: float | None = None,
            color_map: str = 'unknown', skip_duplicates: bool = False) -> None:
        """Add a stimulator after checking that the screen axes are orthogonal."""
        screen_to_field_matrix(screen_x_axis, screen_y_axis)
        self.insert1(dict(
            stimulator_id=stimulator_id, setupid=str(setupid), projection=projection, valid_from=valid_from,
            valid_to=valid_to, documented_from=documented_from, documented_to=documented_to,
            screen_x_axis=screen_x_axis, screen_y_axis=screen_y_axis, offset_relx_um=offset_relx_um,
            offset_rely_um=offset_rely_um, um_per_px=um_per_px, stimulator_delay_ms=stimulator_delay_ms,
            color_map=color_map, confidence=confidence, source=source), skip_duplicates=skip_duplicates)

    def add_defaults(self, skip_duplicates: bool = True) -> None:
        """Add the stimulators of setups 1, 2 and 3: one per period of constant stimulator delay
        (stimulator_delay.csv), with the documented geometry of setups 1 and 3 and the assumed one of setup 2."""
        for row in DEFAULT_STIMULATORS:
            self.add(**row, skip_duplicates=skip_duplicates)

    def get_geometry(self, stimulator_id: str) -> dict:
        """Screen axes and offsets of one stimulator."""
        x_axis, y_axis, offset_relx_um, offset_rely_um = (self & dict(stimulator_id=stimulator_id)).fetch1(
            'screen_x_axis', 'screen_y_axis', 'offset_relx_um', 'offset_rely_um')
        return dict(screen_x_axis=x_axis, screen_y_axis=y_axis,
                    offset_relx_um=float(offset_relx_um), offset_rely_um=float(offset_rely_um))


class ImagingSetupTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        # Imaging parameters of one setup and period
        imaging_setup_id       : varchar(32)    # e.g. 'setup1'
        ---
        setupid                : varchar(191)   # setupid as in Experiment.ExpInfo
        valid_from             : date           # first experiment date these parameters apply to
        valid_to = null        : date           # last experiment date (NULL: open end)
        scan_width_um          : float32        # full-field scan width at zoom 1; pixel size = width / npix / zoom
        npixartifact           : int16          # number of scan lines with the light artifact
        confidence             : enum('high', 'medium', 'low')
        source                 : varchar(511)
        """
        return definition

    def add_defaults(self, skip_duplicates: bool = True) -> None:
        """Add the former setupid constants of djimaging.utils.scanm.setup_utils for setups 1, 2 and 3."""
        self.insert(DEFAULT_IMAGING_SETUPS, skip_duplicates=skip_duplicates)


class _ExperimentSetupAssignmentMixin:
    """Assigns one row of a setup table (stimulator, imaging setup) to each experiment from its setupid and date."""
    _id_key = None  # Primary key of the setup table
    _assign = None  # Function (setupid, date, rows) -> (id, confidence, reason)

    @property
    def _setup_table(self):
        raise NotImplementedError

    @property
    def key_source(self):
        try:
            return self.experiment_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        setupid = (self.experiment_table.ExpInfo & key).fetch1('setupid')
        row_id, confidence, reason = type(self)._assign(setupid, key['date'], self._setup_table().to_dicts())

        if row_id is None:
            warnings.warn(f"Not assigned {key}: {reason}. Add a row to {self._setup_table.__name__} or use add_manual.")
            return

        self.insert1({**key, self._id_key: row_id, 'confidence': confidence, 'reason': reason[:1023]})

    def add_manual(self, key: dict, row_id: str, confidence: str, reason: str) -> None:
        """Assign manually, e.g. for an experiment whose setup or date is wrong in the header."""
        exp_keys = (self.experiment_table & key).keys()
        if len(exp_keys) != 1:
            raise ValueError(f"Expected one experiment for {key}, found {len(exp_keys)}")
        self.insert1({**exp_keys[0], self._id_key: row_id, 'confidence': confidence, 'reason': reason[:1023]},
                     allow_direct_insert=True)

    def report_unassigned(self):
        """Experiments without assignment, with setupid and the reason."""
        df = (self.experiment_table.ExpInfo.proj('setupid') & (self.key_source - self.proj())).to_pandas()
        if len(df) == 0:
            return df
        rows = self._setup_table().to_dicts()
        df = df.reset_index()
        df['reason'] = [type(self)._assign(s, d, rows)[2] for s, d in zip(df['setupid'], df['date'])]
        return df

    def report(self):
        """Number of experiments per assigned row, confidence and reason."""
        df = self.to_pandas().reset_index()
        if len(df) == 0:
            return df
        return df.groupby([self._id_key, 'confidence', 'reason']).size().rename('n_experiments').reset_index()

    def _fetch_assigned(self, key: dict, *attrs):
        rows = (self & key).to_dicts()
        if len(rows) != 1:
            raise ValueError(f"{type(self).__name__}: no assignment for {key}. Populate it (and add_manual for "
                             f"unassigned experiments) before Field and Presentation.")
        values = (self._setup_table & {self._id_key: rows[0][self._id_key]}).fetch1(*attrs)
        return rows[0], values


class ExperimentStimulatorTemplate(_ExperimentSetupAssignmentMixin, dj.Computed):
    database = ""
    _id_key = 'stimulator_id'
    _assign = staticmethod(assign_stimulator)

    @property
    def definition(self):
        definition = """
        # Stimulator of an experiment, from its setup and date
        -> self.experiment_table
        ---
        -> self.stimulator_table
        confidence         : enum('high', 'medium', 'low')  # confidence in the stimulator geometry for this experiment
        reason             : varchar(1023)                   # why the confidence is not high
        """
        return definition

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def stimulator_table(self):
        pass

    @property
    def _setup_table(self):
        return self.stimulator_table

    def get_geometry(self, key: dict) -> dict:
        """Screen axes, offsets, stimulator id, confidence and reason for an experiment (or any key below it)."""
        stimulator_id, confidence, reason = (self & key).fetch1('stimulator_id', 'confidence', 'reason')
        geometry = self.stimulator_table().get_geometry(stimulator_id)
        geometry.update(stimulator_id=stimulator_id, confidence=confidence, reason=reason,
                        screen_to_field=screen_to_field_matrix(geometry['screen_x_axis'], geometry['screen_y_axis']))
        return geometry

    def get_stimulator_delay_s(self, key: dict) -> float:
        """Stimulator delay in seconds for an experiment (or any key below it)."""
        _, delay_ms = self._fetch_assigned(key, 'stimulator_delay_ms')
        return float(delay_ms) / 1000.

    def plot_screen_axes(self, key: dict):
        """Draw where the QDSpy screen axes point in the chamber top view (dorsal/back up for prepwmorient=0)."""
        from matplotlib import pyplot as plt
        geometry = self.get_geometry(key)
        m = geometry['screen_to_field']
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        for col, label, color in [(0, 'screen +x', 'C0'), (1, 'screen +y', 'C1')]:
            dx, dy = m[:, col]  # XCoord (down in top view), YCoord (right)
            ax.annotate('', xy=(dy, -dx), xytext=(0, 0), arrowprops=dict(color=color, width=2))
            ax.text(1.1 * dy, -1.1 * dx, label, color=color, ha='center', va='center')
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect='equal', xticks=[], yticks=[],
               title=f"{geometry['stimulator_id']} ({geometry['confidence']})",
               xlabel='YCoord_um ->', ylabel='<- XCoord_um (curtain down)')
        ax.axhline(0, c='gray', lw=0.5)
        ax.axvline(0, c='gray', lw=0.5)
        return fig, ax


class ExperimentImagingSetupTemplate(_ExperimentSetupAssignmentMixin, dj.Computed):
    database = ""
    _id_key = 'imaging_setup_id'
    _assign = staticmethod(assign_imaging_setup)

    @property
    def definition(self):
        definition = """
        # Imaging setup of an experiment, from its setup and date
        -> self.experiment_table
        ---
        -> self.imaging_setup_table
        confidence         : enum('high', 'medium', 'low')
        reason             : varchar(1023)                   # why the confidence is not high
        """
        return definition

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def imaging_setup_table(self):
        pass

    @property
    def _setup_table(self):
        return self.imaging_setup_table

    def get_recording_params(self, key: dict) -> dict:
        """`scan_width_um` and `npixartifact` for ScanMRecording, for an experiment (or any key below it)."""
        _, (scan_width_um, npixartifact) = self._fetch_assigned(key, 'scan_width_um', 'npixartifact')
        return dict(scan_width_um=float(scan_width_um), npixartifact=int(npixartifact))
