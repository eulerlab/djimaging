from __future__ import annotations

import os
from copy import deepcopy

import datajoint as dj

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.filesystem_utils import print_tree


class UserInfoTemplate(dj.Manual):
    database = ""

    # Multi-channel recordings. If True, the stacks (channels) are defined by `channel_table` instead of
    # `data_stack_name` and `alt_stack_name`, which are then not part of this table.
    # Changes the table definition, so it requires a new/repopulated table.
    _use_channels = False

    @property
    def channel_table(self) -> dj.Table:
        """Return the channel table. Required when `_use_channels` is True."""
        raise NotImplementedError("Set `channel_table` (a ChannelTemplate) when using `_use_channels=True`")

    @property
    def definition(self):
        if self._use_channels:
            stack_names = ""
        else:
            stack_names = """
        data_stack_name='wDataCh0'      :varchar(32)  # main data channel, e.g. OGB-1
        alt_stack_name='wDataCh1'       :varchar(32)  # alternative data channel, e.g. SR101"""

        definition = f"""
        # Info for decoding file names
        experimenter                    :varchar(16)  # name of the experimenter
        ---     
        data_dir                        :varchar(191)   # acquisition-tree root directory used for file discovery
        field_loc                       :int32       # string location for field
        stimulus_loc                    :int32       # string location for stimulus
        animal_loc=NULL                 :int32       # string location for number of animal (e.g. M1)
        datatype_loc=NULL               :int32       # string location for datatype (e.g. SMP)
        region_loc=NULL                 :int32       # string location for region (e.g. LR or RR)
        cond1_loc=NULL                  :int32       # string location for condition 1 (e.g. pharmacological)
        cond2_loc=NULL                  :int32       # string location for condition 2 (e.g. pharmacological)
        cond3_loc=NULL                  :int32       # string location for condition 3 (e.g. pharmacological)
        opticdisk_alias='od_opticdisk_opticdisc'  :varchar(191)  # alias(es) for optic disk (separated by _)
        outline_alias='outline_edge'    :varchar(191)  # alias(es) for retinal outline / edge (separated by _)
        highres_alias='hq_hr_highresolution_512' :varchar(191)  # alias(es) for high resolution stack
        mask_alias='chirp_mb_movingbar' :varchar(191)  # Ordered alias(es) for field roi mask (separated by _)
        pre_data_dir='Pre'              :varchar(32)  # directory for h5 data files
        raw_data_dir='Raw'              :varchar(32)  # directory for smp and smh data files{stack_names}
        """
        return definition

    def get_ch_names(self, key: dict | None = None) -> list[str]:
        """Names of the data stacks (channels). The first is the main channel, e.g. the target of ROI detection.

        With `_use_channels`, the channels of `channel_table` (starting with its `roi_default` channel),
        otherwise `data_stack_name` and `alt_stack_name` of the user in `key`.
        """
        if self._use_channels:
            return self.channel_table().get_ch_names()

        data_name, alt_name = (self & key).fetch1('data_stack_name', 'alt_stack_name')
        return list(dict.fromkeys([data_name, alt_name]))

    def get_stack_names(self, key: dict | None = None) -> tuple[str, str | None]:
        """Names of the main and the alternative data stack. The alternative is None if there is only one channel.

        With several channels, the alternative is the first channel that is not the main channel.
        """
        if self._use_channels:
            ch_names = self.get_ch_names(key)
            return ch_names[0], ch_names[1] if len(ch_names) > 1 else None

        return (self & key).fetch1('data_stack_name', 'alt_stack_name')

    def upload_users(self, userdicts: list) -> None:
        """Upload multiple users to the table.

        Args:
            userdicts: List of user info dicts, each passed to
                ``upload_user``.
        """
        for userdict in userdicts:
            self.upload_user(userdict)

    def upload_user(self, userdict: dict, verbose: int = 1) -> None:
        """Upload a single user entry to the table.

        If the experimenter already exists, the existing entry is preserved and
        a message is printed when verbose > 0.

        Args:
            userdict: Dict containing at minimum 'experimenter' and 'data_dir'
                keys, plus any optional columns defined in the table schema.
            verbose: Verbosity level. If > 0, print a message when the entry
                already exists. Default is 1.

        Raises:
            AssertionError: If 'experimenter' or 'data_dir' are missing, or if
                the data_dir path does not exist on the filesystem.
        """
        assert "experimenter" in userdict, 'Set username'
        assert "data_dir" in userdict, 'Set data_dir'
        assert os.path.isdir(userdict['data_dir']), f"data_dir={userdict['data_dir']} is not a directory"

        userdict = deepcopy(userdict)
        if not userdict['data_dir'].endswith('/'):
            userdict['data_dir'] += '/'

        # Change all aliases to lower case
        for k, v in userdict.items():
            if 'alias' in k:
                userdict[k] = v.lower()

        if userdict["experimenter"] not in self.to_arrays("experimenter"):
            self.insert([userdict])
        else:
            entry = (self & dict(experimenter=userdict["experimenter"])).fetch1()

            if entry != userdict and verbose > 0:
                print(f"Information for `{userdict['experimenter']}` already uploaded.")
                print("If you want to change the user entry, delete the existing one first and upload the user again.")

    def plot1(self, key: dict = None, show_pre: bool = True, show_raw: bool = False, show_header: bool = True) -> None:
        """Plot files available for this user
        :param key: Key to plot.
        :param show_pre: Show files in preprocessed data directory?
        :param show_raw: Show files in raw data directory?
        :param show_header: Show header file names?
        """
        key = get_primary_key(table=self, key=key)

        from djimaging.utils import filesystem_utils
        data_dir = (self & key).fetch1('data_dir')

        assert os.path.isdir(data_dir), f"data_dir={data_dir} is not a directory"

        include_types = []
        if show_pre:
            include_types.append('h5')
        if show_raw:
            include_types.append('smp')
        if show_header:
            include_types.append('ini')

        print_tree(data_dir, include_types=include_types)
