import datajoint as dj
import numpy as np


class StimulusTemplate(dj.Manual):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Light stimuli
        stim_id    :smallint     # Unique integer identifier
        stim_v     :varchar(255) # Stimulus version, e.g. to differentiate between gChirp and lChirp
        ---
        stim_name           :varchar(255)       # string identifier
        alias               :varchar(9999)      # Strings (_ seperator) used to identify this stimulus
        framerate           :float              # framerate in Hz
        isrepeated=0        :tinyint unsigned   # Is the stimulus repeated? Used for snippets
        ntrigger_rep=0      :int unsigned       # Number of triggers per repetition  
        is_colour=0         :tinyint unsigned   # is stimulus coloured (e.g., noise vs. cnoise)
        stim_path=""        :varchar(255)       # Path to hdf5 file containing numerical array and info about stim
        commit_id=""        :varchar(255)       # Commit id corresponding to stimulus entry in Github repo
        """
        return definition

    class StimInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            # General stimulus information for convient inserting of new stimuli
            ->Stimulus
            ---
            trialinfo=NULL      :longblob      # Array of stimulus
            stim_info=NULL      :longblob      # Some data of stimulus, e.g. a dict
            """
            return definition

    def check_alias(self, alias, stim_v, stim_id):
        # Skip duplicate comparison
        existing_aliases = (self - [dict(stim_id=stim_id, stim_v=stim_v)]).fetch('alias')
        for existing_alias in existing_aliases:
            for existing_alias_i in existing_alias.split('_'):
                assert existing_alias_i not in alias.split('_'), \
                    f'Found existing alias `{existing_alias_i}`. Set `unique_alias` to False to insert duplicate.'

    def add_stimulus(self, stim_id, stim_v, stim_name, alias, framerate,
                     isrepeated=0, ntrigger_rep=0, is_colour=0, stim_path="", commit_id="",
                     trialinfo=None, stim_info=None, skip_duplicates=False, unique_alias=True):

        if unique_alias:
            self.check_alias(alias, stim_v, stim_id)

        key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stim_name": stim_name,
            "alias": alias,
            "ntrigger_rep": ntrigger_rep,
            "isrepeated": isrepeated,
            "is_colour": is_colour,
            "framerate": framerate,
            "stim_path": stim_path,
            "commit_id": commit_id,
        }

        self.insert1(key, skip_duplicates=skip_duplicates)

        if trialinfo is not None or stim_info is not None:
            stiminfo_key = {
                "stim_id": stim_id,
                "stim_v": stim_v,
                "trialinfo": trialinfo,
                "stim_info": stim_info,
            }
            self.StimInfo().insert1(stiminfo_key, skip_duplicates=skip_duplicates)

    def add_nostim(self, stim_v="default", alias="nostim_none", skip_duplicates=False):
        """Add none stimulus"""
        self.add_stimulus(
            stim_id=-1,
            stim_v=stim_v,
            stim_name='nostim',
            alias=alias,
            framerate=0,
            skip_duplicates=skip_duplicates,
            unique_alias=True
        )

    class NoiseInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            # noise stimulus
            ->Stimulus
            ---
            pix_n_x              :tinyint        # Number of pixels in stimulus X dimension
            pix_n_y              :tinyint        # Number of pixels in stimulus Y dimension
            pix_scale_x_um       :float          # Length of stimulus in X dimension
            pix_scale_y_um       :float          # Length of stimulus in Y dimension
            trialinfo=NULL       :longblob       # Array of stimulus
            frozencondition=NULL :tinyint        # If noise: Which condition corresponds to frozen noise
            """
            return definition

    def add_noise(self, pix_n_x=20, pix_n_y=15, pix_scale_x_um=30, pix_scale_y_um=30, stim_v=None,
                  key=None, skip_duplicates=False):
        stim_id = 0

        if stim_v is None:
            stim_v = f"{pix_scale_x_um:.0f}um"

        default_key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stim_name": 'noise',
            "alias": f"dn_noise_noise{pix_scale_x_um}m",
            "ntrigger_rep": 1500,
            "isrepeated": 0,
            "is_colour": 0,
            "framerate": 5,
        }

        info = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "pix_n_x": pix_n_x,
            "pix_n_y": pix_n_y,
            "pix_scale_x_um": pix_scale_x_um,
            "pix_scale_y_um": pix_scale_y_um,
        }

        if key is not None:
            default_key.update(key)

        self.check_alias(default_key['alias'], stim_v, stim_id)

        self.insert1(default_key, skip_duplicates=skip_duplicates)
        self.NoiseInfo().insert1(info, skip_duplicates=skip_duplicates)

    class ChirpInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            # local or global chirp stimulus
            ->Stimulus
            ---
            spatialextent       :int        # diameter of stimulus in um
            """
            return definition

    def add_chirp(self, spatialextent, stim_v=None, key=None, skip_duplicates=False):
        stim_id = 1

        if stim_v is None:
            stim_v = f"{spatialextent:.0f}um"

        default_key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stim_name": 'chirp',
            "alias": "chirp_gchirp_globalchirp",
            "is_colour": 0,
            "framerate": 1 / 60.,
            "isrepeated": 1,
            "ntrigger_rep": 2,
        }

        info = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "spatialextent": spatialextent,
        }

        if key is not None:
            default_key.update(key)

        self.check_alias(default_key['alias'], stim_v, stim_id)

        self.insert1(default_key, skip_duplicates=skip_duplicates)
        self.ChirpInfo().insert1(info, skip_duplicates=skip_duplicates)

    class DsInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            # moving bar stimulus
            ->Stimulus
            ---
            trialinfo       :longblob   # 1D array with directions, either for for trial or for all repetions
            bardx           :float      # bar x extension in um
            bardy           :float      # bar y extension in um
            velumsec        :float      # bar movement velocity um/sec
            tmovedurs       :float      # amount of time bar is displayed
            """
            return definition

    def add_movingbar(self, trialinfo=None, bardx=-1, bardy=-1, velumsec=-1, tmovedurs=-1,
                      stim_v="default", key=None, skip_duplicates=False):
        stim_id = 2

        if trialinfo is None:
            trialinfo = np.array([0, 180, 45, 225, 90, 270, 135, 315])

        default_key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stim_name": 'movingbar',
            "alias": "mb_mbar_bar_movingbar",
            "is_colour": 0,
            "framerate": 1 / 60.,            
            "ntrigger_rep": 1,
            "isrepeated": 1,
        }

        info = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "trialinfo": trialinfo,
            "bardx": bardx,
            "bardy": bardy,
            "velumsec": velumsec,
            "tmovedurs": tmovedurs,
        }

        if key is not None:
            default_key.update(key)

        self.check_alias(default_key['alias'], stim_v, stim_id)

        self.insert1(default_key, skip_duplicates=skip_duplicates)
        self.DsInfo().insert1(info, skip_duplicates=skip_duplicates)

