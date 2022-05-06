import os
from copy import deepcopy

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
        highres_alias='hq_hr_highresolution_512' :varchar(255)  # alias(es) for high resolution stack
        """
        return definition

    def upload_users(self, userdicts: list):
        """Upload multiple users"""
        for userdict in userdicts:
            self.upload_user(userdict)

    def upload_user(self, userdict: dict):
        """Upload one user"""
        assert "experimenter" in userdict, 'Set username'
        assert "data_dir" in userdict, 'Set data_dir'
        assert os.path.isdir(userdict['data_dir']), f"data_dir={userdict['data_dir']} is not a directory"

        userdict = deepcopy(userdict)
        if not userdict['data_dir'].endswith('/'):
            userdict['data_dir'] += '/'

        if 'opticdisk_alias' in userdict:
            userdict['opticdisk_alias'] = userdict['opticdisk_alias'].lower()

        if 'outline_alias' in userdict:
            userdict['outline_alias'] = userdict['outline_alias'].lower()

        if 'highres_alias' in userdict:
            userdict['highres_alias'] = userdict['highres_alias'].lower()

        if userdict["experimenter"] not in self.fetch("experimenter"):
            self.insert([userdict])
        else:
            entry = (self & dict(experimenter=userdict["experimenter"])).fetch1()

            if entry != userdict:
                print(f"Information for `{userdict['experimenter']}` already uploaded.")
                print("If you want to change the user entry, delete the existing one first and upload the user again.")

    def plot1(self, key: dict, show_pre: bool = True, show_raw: bool = False, show_header: bool = True) -> None:
        """Plot files available for this user
        :param key: Key to plot.
        :param show_pre: Show files in preprocessed data directory?
        :param show_raw: Show files in raw data directory?
        :param show_header: Show header file names?
        """
        from djimaging.utils import datafile_utils
        data_dir = (self & key).fetch1('data_dir')

        assert os.path.isdir(data_dir), f"data_dir={data_dir} is not a directory"

        include_types = []
        if show_pre:
            include_types.append('h5')
        if show_raw:
            include_types.append('smp')
        if show_header:
            include_types.append('ini')

        datafile_utils.print_tree(data_dir, include_types=include_types)
