import datajoint as dj


class UserInfoTemplate(dj.Manual):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Info for decoding file names
    
        experimenter                :varchar(255)    # name of the experimenter
        ---
        data_dir                    :varchar(255)    # path to header file, used for computed tables
        datatype_loc                :tinyint         # string location for datatype (eg. SMP)
        animal_loc                  :tinyint         # string location for number of animal (e.g. M1)
        region_loc                  :tinyint         # string location for region (eg. LR or RR)
        field_loc                   :tinyint         # string location for field
        stimulus_loc                :tinyint         # string location for stimulus
        condition_loc               :tinyint         # string location for (pharmacological) condition
        pre_data_dir='Pre'          :varchar(255)    # directory for h5 data files
        raw_data_dir='Raw'          :varchar(255)    # directory for raw data files
        data_stack_name='wDataCh0'  :varchar(255)    # name of data stack
        opticdisk_alias='od_opticdisk' :varchar(255) # alias(es) for optic disk recordings (separated by _)
        outline_alias='outline'        :varchar(255) # alias(es) for retinal outline / edge recordings (separated by _)
        """
        return definition

    def upload_user(self, *userdicts):
        for userdict in userdicts:
            assert "experimenter" in userdict, 'Set username'
            if userdict["experimenter"] not in self.fetch("experimenter"):
                self.insert([userdict])
            else:
                entry = (self & dict(experimenter=userdict["experimenter"])).fetch1()
                if entry != userdict:
                    print(f"WARNING: Different information for `{userdict['experimenter']}` already uploaded.")
                else:
                    print(f"Information for `{userdict['experimenter']}` already uploaded")

