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
        stimulusname        :varchar(255)       # string identifier
        alias               :varchar(9999)      # Strings (_ seperator) used to identify this stimulus
        framerate=0         :float              # framerate in hz
        isrepeated=0        :tinyint unsigned   # Is the stimulus repeated? Used for snippets
        ntrigger_rep=0      :int unsigned       # Number of triggers per repetition  
        is_colour=0         :tinyint            # is stimulus coloured (e.g., noise vs. cnoise)
        stim_path=""        :varchar(255)       # Path to hdf5 file containing numerical array and info about stim
        commit_id=""        :varchar(255)       # Commit id corresponding to stimulus entry in Github repo
        """
        return definition

    def add_no_stim(self, stim_v="standard", key=None, skip_duplicates=False):
        """Add none stimulus"""
        stim_id = -1

        default_key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stimulusname": 'no_stim',
            "alias": 'nostim_none',
        }

        if key is not None:
            default_key.update(key)

        self.insert1(default_key, skip_duplicates=skip_duplicates)

    class StimInfo(dj.Part):
        definition = """
        # General stimulus information for convient inserting of new stimuli
        ->Stimulus
        ---
        trialinfo=NULL          :longblob      # Array of stimulus
        stimulus_info=NULL      :longblob      # Some data of stimulus, de
        stimulus_description="" :varchar(9999) # additional stimulus info in string format
        """

    def add_stimulus(self, stim_id, stim_v, stimulusname, alias, ntrigger_rep, isrepeated, is_colour, framerate,
                     trialinfo=None, stimulus_data=None, stimulus_info=None,
                     key=None, skip_duplicates=False):

        default_key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stimulusname": stimulusname,
            "alias": alias,
            "ntrigger_rep": ntrigger_rep,
            "isrepeated": isrepeated,
            "is_colour": is_colour,
            "framerate": framerate,
        }

        info = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "trialinfo": trialinfo,
            "stimulus_data": stimulus_data,
            "stimulus_info": stimulus_info,
        }

        if key is not None:
            default_key.update(key)

        self.insert1(default_key, skip_duplicates=skip_duplicates)
        self.StimInfo().insert1(info, skip_duplicates=skip_duplicates)

    class NoiseInfo(dj.Part):
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

    def add_noise(self, pix_n_x=20, pix_n_y=15, pix_scale_x_um=30, pix_scale_y_um=30, stim_v=None,
                  key=None, skip_duplicates=False):
        stim_id = 0

        if stim_v is None:
            stim_v = f"{pix_scale_x_um:.0f}um"

        default_key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stimulusname": 'noise',
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

        self.insert1(default_key, skip_duplicates=skip_duplicates)
        self.NoiseInfo().insert1(info, skip_duplicates=skip_duplicates)

    class ChirpInfo(dj.Part):
        definition = """
        # local or global chirp stimulus
        ->Stimulus
        ---
        spatialextent       :int        # diameter of stimulus in um
        """

    def add_chirp(self, spatialextent, stim_v=None, key=None, skip_duplicates=False):
        stim_id = 1

        if stim_v is None:
            stim_v = f"{spatialextent:.0f}um"

        default_key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stimulusname": 'chirp',
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

        self.insert1(default_key, skip_duplicates=skip_duplicates)
        self.ChirpInfo().insert1(info, skip_duplicates=skip_duplicates)

    class DsInfo(dj.Part):
        definition = """
        # moving bar stimulus
        ->Stimulus

        ---
        trialinfo       :longblob   # 1D array with directions, either for for trial or for all repetions
        bardx           :int        # bar x extension in um
        bardy           :int        # bar y extension in um
        velumsec        :float      # bar movement velocity um/sec
        tmovedurs       :float      # amount of time bar is displayed
        """

    def add_movingbar(self, trial_info=None, stim_v="default", key=None, skip_duplicates=False):
        stim_id = 2

        if trial_info is None:
            trial_info = np.array([0, 180, 45, 225, 90, 270, 135, 315])

        default_key = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "stimulusname": 'movingbar',
            "alias": "mb_mbar_bar_movingbar",
            "is_colour": 0,
            "framerate": 1 / 60.,            
            "ntrigger_rep": 1,
            "isrepeated": 1,
        }

        info = {
            "stim_id": stim_id,
            "stim_v": stim_v,
            "trial_info": trial_info,
        }

        if key is not None:
            default_key.update(key)

        self.insert1(default_key, skip_duplicates=skip_duplicates)
        self.DsInfo().insert1(info, skip_duplicates=skip_duplicates)

