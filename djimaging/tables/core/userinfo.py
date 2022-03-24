import os

import datajoint as dj


class UserInfoTemplate(dj.Manual):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Info for decoding file names
    
        experimenter                   :varchar(255)  # name of the experimenter
        ---    
        data_dir                       :varchar(255)  # path to header file, used for computed tables
        datatype_loc                   :tinyint       # string location for datatype (eg. SMP)
        animal_loc                     :tinyint       # string location for number of animal (e.g. M1)
        region_loc                     :tinyint       # string location for region (eg. LR or RR)
        field_loc                      :tinyint       # string location for field
        stimulus_loc                   :tinyint       # string location for stimulus
        condition_loc                  :tinyint       # string location for (pharmacological) condition
        pre_data_dir='Pre'             :varchar(255)  # directory for h5 data files
        raw_data_dir='Raw'             :varchar(255)  # directory for raw data files
        data_stack_name='wDataCh0'     :varchar(255)  # name of data stack
        opticdisk_alias='od_opticdisk' :varchar(255)  # alias(es) for optic disk recordings (separated by _)
        outline_alias='outline_edge'   :varchar(255)  # alias(es) for retinal outline / edge recordings (separated by _)
        """
        return definition

    def upload_user(self, *userdicts):
        """Upload one or multiple user dicts"""
        for userdict in userdicts:
            assert "experimenter" in userdict, 'Set username'
            if userdict["experimenter"] not in self.fetch("experimenter"):
                self.insert([userdict])
            else:
                entry = (self & dict(experimenter=userdict["experimenter"])).fetch1()
                if entry != userdict:
                    print(f"WARNING: Information for `{userdict['experimenter']}` already uploaded.")
                else:
                    print(f"Information for `{userdict['experimenter']}` already uploaded")

    def plot1(self, key, show_pre=True, show_raw=False, show_header=True):
        """Plot files available for this user"""
        from djimaging.utils import datafile_utils
        data_dir = (self & key).fetch1('data_dir')

        assert os.path.isdir(data_dir), f'ERROR: {data_dir} is not a directory'

        include_types = []
        if show_pre:
            include_types.append('h5')
        if show_raw:
            include_types.append('smp')
        if show_header:
            include_types.append('ini')

        datafile_utils.print_tree(data_dir, include_types=include_types)
