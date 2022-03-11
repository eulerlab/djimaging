import datajoint as dj


class UserInfo(dj.Manual):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Info for decoding file names
    
        experimenter                :varchar(255)             # name of the experimenter
        ---
        data_dir                    :varchar(255)             # path to header file, used for computed tables
        pre_data_dir='Pre'          :varchar(255)             # directory for h5 data files
        raw_data_dir='Raw'          :varchar(255)             # directory for raw data files
        datatype_loc                :tinyint                  # string location for datatype (eg. SMP)
        animal_loc                  :tinyint                  # string location for number of animal (e.g. M1)
        region_loc                  :tinyint                  # string location for region (eg. LR or RR)
        field_loc                   :tinyint                  # string location for field
        stimulus_loc                :tinyint                  # string location for stimulus
        pharm_loc                   :tinyint                  # string location for pharmacology
        """
        return definition

    def upload_user(self, userdict=None):
        uploaded = self.fetch("experimenter")
        for item in userdict:
            if item["experimenter"] not in uploaded:
                self.insert([item])  # more than one user can be uploaded at a time
            else:
                print("Information for that user already uploaded")
