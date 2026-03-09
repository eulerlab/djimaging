from __future__ import annotations

import warnings

import datajoint as dj
import numpy as np

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def reformat_numerical_trial_info(trial_info: list) -> list:
    """Change old list format to new list[dict] format.

    Args:
        trial_info: List of numerical trial identifiers (e.g. direction angles).

    Returns:
        A list of dicts with keys 'name' and 'ntrigger' (always 1) derived
        from the input list.
    """
    return [dict(name=trial_info_i, ntrigger=1) for i, trial_info_i in enumerate(trial_info)]


def check_trial_info(trial_info: list, ntrigger_rep: int) -> list:
    """Validate trial info and normalise to the list[dict] format if necessary.

    Args:
        trial_info: Trial information as either a list of dicts (each with
            'name' and 'ntrigger' keys) or a legacy list of numerical values.
        ntrigger_rep: Expected total number of triggers per repetition. Must
            equal the sum of 'ntrigger' values across all trial_info entries.

    Returns:
        Trial info in the canonical list[dict] format.

    Raises:
        TypeError: If trial_info is not a list or array.
        AssertionError: If an unknown key is present in a trial_info entry or
            if a trigger count is not an integer.
        ValueError: If the trigger count sum does not match ntrigger_rep (only
            raised when ntrigger_rep > 1).
    """
    if not isinstance(trial_info, (list, np.ndarray)):
        raise TypeError('trial_info must either be a list or an array')

    if not isinstance(trial_info[0], dict):
        trial_info = reformat_numerical_trial_info(trial_info)

    for trial_info_i in trial_info:
        for k, v in trial_info_i.items():
            assert k in ['name', 'ntrigger', 'ntrigger_split'], f'Unknown key in trial_info k={k}'

            if k in ['ntrigger', 'ntrigger_split']:
                assert int(v) == v, f'Value for k={k} but be an integer but is v={v}'

    ntrigger_ti = sum([trial_info_i["ntrigger"] for trial_info_i in trial_info])
    if not (ntrigger_ti == ntrigger_rep):
        msg = f'Number of triggers in trial_info={ntrigger_ti} must match ntrigger_rep={ntrigger_rep}.'
        if ntrigger_rep > 1:
            raise ValueError(msg)
        else:
            # Raise only warnings for previous work-around solution
            warnings.warn(msg)

    return trial_info


class StimulusTemplate(dj.Manual):
    database = ""
    _incl_snippet_base_dt = False  # Optional for compatibility with previous versions

    @property
    def definition(self):
        definition = """
        # Light stimuli
        stim_name           :varchar(32)       # Unique string identifier
        ---
        alias               :varchar(999)       # Strings (_ seperator) to identify this stimulus, not case sensitive!
        stim_family=""      :varchar(191)       # To group stimuli (e.g. gChirp and lChirp) for downstream processing 
        framerate=0         :float              # framerate in Hz
        isrepeated=0        :tinyint unsigned   # Is the stimulus repeated? Used for snippets
        ntrigger_rep=0      :mediumint unsigned # Number of triggers (per repetition)
        stim_path=""        :varchar(191)       # Path to hdf5 file containing numerical array and info about stim
        commit_id=""        :varchar(191)       # Commit id corresponding to stimulus entry in GitHub repo
        stim_hash=""        :varchar(191)       # QDSpy hash
        trial_info=NULL     :longblob           # trial information, e.g. directions of moving bar
        stim_trace=NULL     :longblob           # array of stimulus if available
        stim_dict=NULL      :longblob           # stimulus information dictionary, contains e.g. spatial extent
        """

        if self._incl_snippet_base_dt:
            definition += """
            snippet_base_dt=NULL : float           # Time used for snippet baseline estimation
            """

        return definition

    def check_alias(self, alias: str, stim_name: str) -> None:
        """Raise an error if any part of alias is already used by another stimulus.

        Args:
            alias: Underscore-separated alias string for the new stimulus.
            stim_name: Name of the stimulus being added (excluded from the
                duplicate check so re-inserting the same stimulus is allowed).

        Raises:
            AssertionError: If any individual alias token already exists in the
                table for a different stimulus.
        """
        existing_aliases = (self - [dict(stim_name=stim_name)]).fetch('alias')  # Skip duplicate comparison
        for existing_alias in existing_aliases:
            for existing_alias_i in existing_alias.split('_'):
                assert existing_alias_i not in alias.split('_'), \
                    f'Found existing alias `{existing_alias_i}`. Set `unique_alias` to False to insert duplicate.'

    def add_stimulus(self, stim_name: str, alias: str, stim_family: str = "", framerate: float = 0,
                     isrepeated: bool = 0, ntrigger_rep: int = 0, snippet_base_dt: float = None,
                     stim_path: str = "", commit_id: str = "",
                     trial_info: Iterable = None, stim_trace: np.ndarray = None, stim_dict: dict = None,
                     skip_duplicates: bool = False, unique_alias: bool = True) -> None:
        """
        Add stimulus to database
        :param stim_name: See table definition.
        :param alias: See table definition.
        :param stim_family: See table definition.
        :param framerate: See table definition.
        :param isrepeated: See table definition.
        :param ntrigger_rep: See table definition.
        :param snippet_base_dt: See table definition.
        :param stim_path: See table definition.
        :param commit_id: See table definition.
        :param trial_info: See table definition.
        :param stim_trace: See table definition.
        :param stim_dict: See table definition.
        :param skip_duplicates: Silently skip duplicates.
        :param unique_alias: Check if any of the aliases is already in use.
        """
        assert np.round(ntrigger_rep) == ntrigger_rep, 'Value needs to be an integer'
        assert np.round(isrepeated) == isrepeated, 'Value needs to be an integer'

        if unique_alias:
            self.check_alias(alias, stim_name=stim_name)

        if stim_dict is not None:
            missing_info = [k for k, v in stim_dict.items() if v is None]
            if len(missing_info) > 0:
                warnings.warn(f'Values for {missing_info} in `stim_dict` for stimulus `{stim_name}` are None. '
                              + 'This may cause problems downstream.')

        if trial_info is not None:
            # noinspection PyTypeChecker
            check_trial_info(trial_info=trial_info, ntrigger_rep=ntrigger_rep)

        key = {
            "stim_name": stim_name,
            "alias": alias.lower(),
            "stim_family": stim_family,
            "ntrigger_rep": int(np.round(ntrigger_rep)),
            "isrepeated": int(np.round(isrepeated)),
            "framerate": framerate,
            "stim_path": stim_path,
            "commit_id": commit_id,
            "trial_info": trial_info,
            "stim_trace": stim_trace,
            "stim_dict": stim_dict,
        }

        if self._incl_snippet_base_dt:
            key["snippet_base_dt"] = snippet_base_dt
        elif snippet_base_dt is not None:
            raise ValueError(
                'Snippet base dt is not supported for this table. '
                'Set `_incl_snippet_base_dt` to True in the table definition or remove the parameter.'
            )

        self.insert1(key, skip_duplicates=skip_duplicates)

    def update_trial_info_format(self, restriction: dict = None) -> None:
        """Update all trial info entries to the list[dict] format.

        Note: This modifies existing rows in-place and breaks compatibility
        with code that expects the old numerical list format.

        Args:
            restriction: Optional restriction dict applied before fetching
                keys. Defaults to no restriction (all entries).
        """
        if restriction is None:
            restriction = dict()

        for key in (self & restriction).proj().fetch(as_dict=True):
            trial_info, ntrigger_rep = (self & key).fetch1('trial_info', 'ntrigger_rep')
            if trial_info is not None:
                trial_info = check_trial_info(trial_info=trial_info, ntrigger_rep=ntrigger_rep)
                self.update1(dict(**key, trial_info=trial_info))

    def add_nostim(self, alias: str = "nostim_none", skip_duplicates: bool = False) -> None:
        """Add a placeholder 'no stimulus' entry to the table.

        Args:
            alias: Underscore-separated alias string for the nostim entry.
                Default is 'nostim_none'.
            skip_duplicates: If True, silently skip if the entry already
                exists. Default is False.
        """
        self.add_stimulus(
            stim_name='nostim',
            alias=alias,
            framerate=0,
            skip_duplicates=skip_duplicates,
            unique_alias=True
        )

    def add_noise(
            self,
            stim_name: str = "noise",
            stim_family: str = 'noise',
            framerate: float = 5.,
            ntrigger_rep: int = 1500,
            isrepeated: bool = False,
            snippet_base_dt: float = None,
            alias: str = None,
            ntrigger_per_frame: int = 1,
            stim_trace: np.ndarray = None,
            pix_n_x: int = None,
            pix_n_y: int = None,
            pix_scale_x_um: float = None,
            pix_scale_y_um: float = None,
            n_colors: int = None,
            locations_cats: list = None,
            offset_x_um: float = None,
            offset_y_um: float = None,
            skip_duplicates: bool = False,
    ) -> None:
        """Add a dense noise stimulus entry to the table.

        Args:
            stim_name: Unique string identifier for this stimulus. Default is
                'noise'.
            stim_family: Stimulus family label. Default is 'noise'.
            framerate: Stimulus frame rate in Hz. Default is 5.
            ntrigger_rep: Number of triggers per repetition. Default is 1500.
            isrepeated: Whether the stimulus is repeated. Default is False.
            snippet_base_dt: Baseline window duration for snippet correction.
            alias: Custom alias string; auto-generated from pixel scale if None.
            ntrigger_per_frame: Number of triggers sent per stimulus frame.
                Default is 1.
            stim_trace: Optional numerical stimulus trace array.
            pix_n_x: Number of stimulus pixels in x.
            pix_n_y: Number of stimulus pixels in y.
            pix_scale_x_um: Pixel scale in x (micrometers).
            pix_scale_y_um: Pixel scale in y (micrometers).
            n_colors: Number of colour channels in the stimulus.
            locations_cats: List of spatial location categories.
            offset_x_um: Stimulus offset in x (micrometers). If None, 0 is
                assumed and a warning is issued.
            offset_y_um: Stimulus offset in y (micrometers). If None, 0 is
                assumed and a warning is issued.
            skip_duplicates: If True, silently skip duplicate entries.
                Default is False.
        """

        if alias is None:
            alias = f"dn_noise_dn{pix_scale_x_um}m_noise{pix_scale_x_um}m"

        stim_dict = {
            "ntrigger_per_frame": ntrigger_per_frame,
        }

        if n_colors is not None:
            stim_dict["n_colors"] = n_colors
        if pix_n_x is not None:
            stim_dict["pix_n_x"] = pix_n_x
        if pix_n_y is not None:
            stim_dict["pix_n_y"] = pix_n_y
        if pix_scale_x_um is not None:
            stim_dict["pix_scale_x_um"] = pix_scale_x_um
        if pix_scale_y_um is not None:
            stim_dict["pix_scale_y_um"] = pix_scale_y_um
        if locations_cats is not None:
            stim_dict["locations_cats"] = locations_cats
        if offset_x_um is not None:
            stim_dict["offset_x_um"] = offset_x_um
        else:
            warnings.warn(
                "Stimulus offset not set. Assuming 0 offset. "
                "This is incorrect for the standard dense noise stimulus.")
            stim_dict["offset_x_um"] = 0

        if offset_y_um is not None:
            stim_dict["offset_y_um"] = offset_y_um
        else:
            warnings.warn(
                "Stimulus offset not set. Assuming 0 offset. "
                "This is incorrect for the standard dense noise stimulus.")
            stim_dict["offset_y_um"] = 0

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            ntrigger_rep=ntrigger_rep,
            isrepeated=isrepeated,
            snippet_base_dt=snippet_base_dt,
            framerate=framerate,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
            stim_trace=stim_trace,
            stim_dict=stim_dict,
        )

    def add_chirp(
            self,
            stim_name: str = "chirp",
            stim_family: str = 'chirp',
            spatialextent: float = None,
            framerate: float = 1 / 60.,
            ntrigger_rep: int = 2,
            isrepeated: bool = True,
            snippet_base_dt: float = None,
            stim_trace: np.ndarray = None,
            alias: str = None,
            skip_duplicates: bool = False,
    ) -> None:
        """Add a chirp stimulus entry to the table.

        Args:
            stim_name: Unique string identifier. Default is 'chirp'.
            stim_family: Stimulus family label. Default is 'chirp'.
            spatialextent: Spatial extent of the stimulus in micrometers.
            framerate: Stimulus frame rate in Hz. Default is 1/60.
            ntrigger_rep: Number of triggers per repetition. Default is 2.
            isrepeated: Whether the stimulus is repeated. Default is True.
            snippet_base_dt: Baseline window duration for snippet correction.
            stim_trace: Optional numerical stimulus trace array.
            alias: Custom alias string; defaults to a standard chirp alias if
                None.
            skip_duplicates: If True, silently skip duplicate entries.
                Default is False.
        """

        if alias is None:
            alias = "chirp_gchirp_globalchirp_lchirp_localchirp"

        stim_dict = {
            "spatialextent": spatialextent,
        }

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            isrepeated=isrepeated,
            ntrigger_rep=ntrigger_rep,
            snippet_base_dt=snippet_base_dt,
            framerate=framerate,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
            stim_trace=stim_trace,
            stim_dict=stim_dict,
        )

    def add_movingbar(
            self,
            stim_name: str = "movingbar",
            stim_family: str = 'movingbar',
            ntrigger_rep: int = 1,
            isrepeated: bool = True,
            trial_info: list = None,
            framerate: float = 1 / 60.,
            snippet_base_dt: float = None,
            bardx: float = None,
            bardy: float = None,
            velumsec: float = None,
            tmovedurs: float = None,
            stim_dict_other: dict = None,
            alias: str = None,
            skip_duplicates: bool = False,
    ) -> None:
        """Add a moving bar stimulus entry to the table.

        Args:
            stim_name: Unique string identifier. Default is 'movingbar'.
            stim_family: Stimulus family label. Default is 'movingbar'.
            ntrigger_rep: Number of triggers per repetition. Default is 1.
            isrepeated: Whether the stimulus is repeated. Default is True.
            trial_info: List of trial dicts (each with 'name' and 'ntrigger').
                If None, defaults to the standard 8-direction protocol
                [0, 180, 45, 225, 90, 270, 135, 315].
            framerate: Stimulus frame rate in Hz. Default is 1/60.
            snippet_base_dt: Baseline window duration for snippet correction.
            bardx: Bar width in micrometers.
            bardy: Bar height in micrometers.
            velumsec: Bar velocity in micrometers per second.
            tmovedurs: Movement duration in seconds.
            stim_dict_other: Additional key-value pairs to merge into the
                stimulus dict.
            alias: Custom alias string; defaults to a standard moving-bar alias
                if None.
            skip_duplicates: If True, silently skip duplicate entries.
                Default is False.
        """

        if trial_info is None:
            trial_info = np.array([0, 180, 45, 225, 90, 270, 135, 315])
            # ===============================================================================
            # Version "a574d90497d97ff9b93381f6b2eb207d"
            # Example users:
            # - Setup-1/Dominic/RGC_MovingBar.py (also used by Tom, Florentyna, Nadine, Ryan, Chenchen, "Colored noise/Klaudia")
            # Directions in script = [0, 180, 45, 225, 90, 270, 135, 315] resulting in:
            # - local QDSpy:       ↓, ↑, ↙, ↗, ←, →, ↖, ↘
            # - played on Setup 1: →, ←, ↗, ↙, ↑, ↓, ↖, ↘ (rotated 90° counterclockwise and then flipped upside-down)
            # - played on Setup 3: →, ←, ↘, ↖, ↓, ↑, ↙, ↗ (rotated 90° counterclockwise)
            # ===============================================================================
            # Version "00bdaae87a8a50500c2063a68bfb3c1e"
            # Example users:
            # - Setup-1/Katrin/RGCs/DS.py
            # - Setup-3/Katrin/RGCs/DS.py
            # Directions in script = [0, 180, 45, 225, 90, 270, 135, 315] resulting in:
            # - local QDSpy:       ←, →, ↖, ↘, ↑, ↓, ↗, ↙
            # - played on Setup 1: ↑, ↓, ↖, ↘, ←, →, ↙, ↗ (rotated 90° counterclockwise and then flipped upside-down)
            # - played on Setup 3: ↓, ↑, ↙, ↗, ←, →, ↖, ↘ (rotated 90° counterclockwise)
            # ===============================================================================
            # Version "bc083a0411701c4141e7b55f6a9c898b"
            # Example users:
            # - Setup-3/Chenchen/RGC_MovingBar.py
            # Directions in script = [0, 180, 45, 225, 90, 270, 135, 315] resulting in:
            # ???
            # ===============================================================================
            # Version "3913b988eb2bcc55e363d1d9d3dcbea9"
            # Example users:
            # - Setup-3/Klaudia/smaller stim size RGCs/RGC_MovingBar
            # Directions in script = [0, 180, 45, 225, 90, 270, 135, 315] resulting in:
            # ???
            # ===============================================================================
            # Version "???"
            # Example users:
            # - Setup-2/Klaudia/RGC_MovingBar???
            # Directions in script = ??? resulting in:
            # ???
            # ===============================================================================
            # Version "???"
            # Example users:
            # - Setup-2/Katrin/RGC_MovingBar???
            # Directions in script = ??? resulting in:
            # ???

        if alias is None:
            alias = "mb_mbar_bar_movingbar"

        stim_dict = {
            "bardx": bardx,
            "bardy": bardy,
            "velumsec": velumsec,
            "tmovedurs": tmovedurs,
        }
        if stim_dict_other is not None:
            stim_dict.update(stim_dict_other)

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            ntrigger_rep=ntrigger_rep,
            isrepeated=isrepeated,
            snippet_base_dt=snippet_base_dt,
            framerate=framerate,
            trial_info=trial_info,
            stim_dict=stim_dict,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
        )
