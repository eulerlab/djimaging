import datajoint as dj


class StimulusTemplate(dj.Manual):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Light stimuli
        stim_id             :tinyint            # Unique integer identifier
        stim_v              :tinyint            # Stimulus version, e.g. die differentiate between gChirp and lChirp
        ---
        framerate           :float              # framerate in hz
        stimulusname        :varchar(255)       # string identifier
        trial_info=NULL     :longblob           # 1d, 2d or 3d array of the stimulus trial info
        stimulus_trace=NULL :longblob           # 2d or 3d array of the stimulus
        stimulus_info=""    :varchar(9999)      # additional stimulus info in string format
        is_colour           :tinyint            # is stimulus coloured (e.g., noise vs. cnoise)
        stim_path           :varchar(255)       # Path to hdf5 file containing numerical array and info about stim
        commit_id           :varchar(255)       # Commit id corresponding to stimulus entry in Github repo
        alias               :varchar(9999)      # Strings (_ seperator) used to identify this stimulus
        isrepeated          :tinyint unsigned   # Is the stimilus repeated? Used for snippets
        ntrigger_rep=0      :int unsigned       # Number of triggers per repetition  
        """
        return definition

