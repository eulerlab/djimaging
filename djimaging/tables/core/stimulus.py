import datajoint as dj
import numpy as np


class StimulusTemplate(dj.Manual):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Light stimuli
        stim_name           :varchar(255)       # Unique string identifier
        ---
        alias               :varchar(9999)      # Strings (_ seperator) to identify this stimulus, not case sensitive!
        stim_family=""      :varchar(255)       # To group stimuli (e.g. gChirp and lChirp) for downstream processing 
        framerate=0         :float              # framerate in Hz
        isrepeated=0        :tinyint unsigned   # Is the stimulus repeated? Used for snippets
        ntrigger_rep=0      :int unsigned       # Number of triggers (per repetition)  
        stim_path=""        :varchar(255)       # Path to hdf5 file containing numerical array and info about stim
        commit_id=""        :varchar(255)       # Commit id corresponding to stimulus entry in GitHub repo
        stim_hash=""        :varchar(255)       # QDSPy hash
        trial_info=NULL     :longblob           # trial information, e.g. directions of moving bar
        stim_trace=NULL     :longblob           # array of stimulus if available
        stim_dict=NULL      :longblob           # stimulus information dictionary, contains e.g. spatial extent
        """
        return definition

    def check_alias(self, alias: str, stim_name: str):
        existing_aliases = (self - [dict(stim_name=stim_name)]).fetch('alias')  # Skip duplicate comparison
        for existing_alias in existing_aliases:
            for existing_alias_i in existing_alias.split('_'):
                assert existing_alias_i not in alias.split('_'), \
                    f'Found existing alias `{existing_alias_i}`. Set `unique_alias` to False to insert duplicate.'

    def add_stimulus(self, stim_name: str, alias: str, stim_family: str = "", framerate: float = 0,
                     isrepeated: bool = 0, ntrigger_rep: int = 0, stim_path: str = "", commit_id: str = "",
                     trial_info: object = None, stim_trace: object = None, stim_dict: dict = None,
                     skip_duplicates: bool = False, unique_alias: bool = True) -> None:
        """
        Add stimulus to database
        :param stim_name: See table defintion.
        :param alias: See table defintion.
        :param stim_family: See table defintion.
        :param framerate: See table defintion.
        :param isrepeated: See table defintion.
        :param ntrigger_rep: See table defintion.
        :param stim_path: See table defintion.
        :param commit_id: See table defintion.
        :param trial_info: See table defintion.
        :param stim_trace: See table defintion.
        :param stim_dict: See table defintion.
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
                print(f'WARNING: Values for {missing_info} in `stim_dict` for stimulus `{stim_name}` are None. '
                      + 'This may cause problems downstream.')

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

        self.insert1(key, skip_duplicates=skip_duplicates)

    def add_nostim(self, alias="nostim_none", skip_duplicates=False):
        """Add none stimulus"""
        self.add_stimulus(
            stim_name='nostim',
            alias=alias,
            framerate=0,
            skip_duplicates=skip_duplicates,
            unique_alias=True
        )

    def add_noise(self, stim_name: str = "noise", stim_family: str = 'noise',
                  framerate: float = 5., ntrigger_rep: int = 1500, isrepeated: bool = False,
                  alias: str = None, pix_n_x: int = None, pix_n_y: int = None,
                  pix_scale_x_um: float = None, pix_scale_y_um: float = None, skip_duplicates: bool = False) -> None:

        if alias is None:
            alias = f"dn_noise_dn{pix_scale_x_um}m_noise{pix_scale_x_um}m"

        stim_dict = {
            "pix_n_x": pix_n_x,
            "pix_n_y": pix_n_y,
            "pix_scale_x_um": pix_scale_x_um,
            "pix_scale_y_um": pix_scale_y_um,
        }

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            ntrigger_rep=ntrigger_rep,
            isrepeated=isrepeated,
            framerate=framerate,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
            stim_dict=stim_dict,
        )

    def add_chirp(self, stim_name: str = "chirp", stim_family: str = 'chirp',
                  spatialextent: float = None, framerate: float = 1 / 60.,
                  ntrigger_rep: int = 2, isrepeated: bool = True,
                  alias: str = None, skip_duplicates: bool = False):

        if alias is None:
            alias = "chirp_gchirp_globalchirp_lchirp_localchirp"

        stim_dict = {
            "spatialextent": spatialextent,
        }

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            ntrigger_rep=ntrigger_rep,
            isrepeated=isrepeated,
            framerate=framerate,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
            stim_dict=stim_dict,
        )

    def add_movingbar(self, stim_name: str = "movingbar", stim_family: str = 'movingbar',
                      bardx: float = None, bardy: float = None, velumsec: float = None, tmovedurs: float = None,
                      ntrigger_rep: int = 1, isrepeated: bool = 1, trial_info=None, framerate: float = 1 / 60.,
                      alias: str = None, skip_duplicates: bool = False):

        if trial_info is None:
            trial_info = np.array([0, 180, 45, 225, 90, 270, 135, 315])

        if alias is None:
            alias = "mb_mbar_bar_movingbar"

        stim_dict = {
            "bardx": bardx,
            "bardy": bardy,
            "velumsec": velumsec,
            "tmovedurs": tmovedurs,
        }

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            ntrigger_rep=ntrigger_rep,
            isrepeated=isrepeated,
            framerate=framerate,
            trial_info=trial_info,
            stim_dict=stim_dict,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
        )
