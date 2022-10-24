import os
import warnings

import datajoint as dj
import numpy as np
import pandas as pd

from djimaging.utils.data_utils import load_h5_data
from djimaging.utils.datafile_utils import find_folders_with_file_of_type
from djimaging.utils.dj_utils import PlaceholderTable
from djimaging.tables.morphology.morphology_utils import get_linestack, compute_df_paths_and_density_maps, \
    compute_density_map_extent, compute_density_center


class StratificationTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Morphfile for experiment assumes single cell per exp_num
        -> self.field_table
        ---
        fromfile : varchar(255)
        line_stratification_xz = NULL : longblob
        line_stratification_yz = NULL : longblob
        """
        return definition

    field_table = PlaceholderTable
    experiment_table = PlaceholderTable

    morph_folder = 'Morph'

    @property
    def key_source(self):
        return self.field_table.Zstack()

    def make(self, key):
        fromfile = (self.field_table() & key).fetch1('fromfile')
        header_path = (self.experiment_table() & key).fetch1('header_path')

        morph_data = load_h5_data(fromfile, lower_keys=True)

        if 'line_stratification_xz' not in morph_data:
            morph_dir = os.path.join(header_path, self.morph_folder)
            morph_file = [f for f in os.listdir(morph_dir) if f.endswith('h5')]
            assert len(morph_file) == 1, f'Found multiple Morph files for key={key}'
            morph_file = morph_file[0]

            fromfile = os.path.join(morph_dir, morph_file)
            morph_data = load_h5_data(fromfile, lower_keys=True)

        line_stratification_xz = morph_data.get('line_stratification_xz', None)
        line_stratification_yz = morph_data.get('line_stratification_yz', None)

        if line_stratification_xz is None and line_stratification_yz is None:
            warnings.warn(f'Did not find stratification for key={key}')
            return

        morph_key = key.copy()
        morph_key['fromfile'] = fromfile
        morph_key['line_stratification_xz'] = line_stratification_xz
        morph_key['line_stratification_yz'] = line_stratification_yz
        self.insert1(morph_key)


class IPLTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Morphfile for experiment assumes single cell per exp_num
        -> self.field_table
        ---
        fromfile : varchar(255)
        scaled_ipl_depth : longblob
        """
        return definition

    field_table = PlaceholderTable
    experiment_table = PlaceholderTable

    morph_folder = 'Morph'

    @property
    def key_source(self):
        return self.field_table.Zstack()

    def make(self, key):
        fromfile = (self.field_table() & key).fetch1('fromfile')
        header_path = (self.experiment_table() & key).fetch1('header_path')

        morph_data = load_h5_data(fromfile, lower_keys=True)

        if 'scaledipldepth' not in morph_data:
            morph_dir = os.path.join(header_path, self.morph_folder)
            morph_file = [f for f in os.listdir(morph_dir) if f.endswith('h5')]
            assert len(morph_file) == 1, f'Found multiple Morph files for key={key}'
            morph_file = morph_file[0]

            fromfile = os.path.join(morph_dir, morph_file)
            morph_data = load_h5_data(fromfile, lower_keys=True)

        scaled_ipl_depth = morph_data.get('scaledipldepth', None)

        if scaled_ipl_depth is None:
            warnings.warn(f'Did not find scaled_ipl_depth for key={key}')
            return

        morph_key = key.copy()
        morph_key['fromfile'] = fromfile
        morph_key['scaled_ipl_depth'] = scaled_ipl_depth

        self.insert1(morph_key)


class SWCTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # SWC for experiment assumes single cell per exp_num
        -> self.field_table
        ---
        swc_path : varchar(255)
        """
        return definition

    field_table = PlaceholderTable
    experiment_table = PlaceholderTable

    @property
    def key_source(self):
        return self.field_table.Zstack()

    def make(self, key):
        header_path = (self.experiment_table() & key).fetch1('header_path')

        swc_folder = find_folders_with_file_of_type(header_path, 'swc')
        assert len(swc_folder) <= 1, f'Found multiple SWC folders for key={key}'
        if len(swc_folder) == 0:
            warnings.warn(f'Did not find swc file for key={key}')
            return
        swc_folder = swc_folder[0]

        swc_file = [f for f in os.listdir(swc_folder) if f.endswith('swc')]
        assert len(swc_file) == 1, f'Found multiple SWC files for key={key}'
        swc_file = swc_file[0]

        swc_key = key.copy()
        swc_key['swc_path'] = os.path.join(swc_folder, swc_file)
        self.insert1(swc_key)


class MorphPathsTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # SWC for experiment assumes single cell per exp_num
        -> self.swc_table
        ---
        soma_xyz: blob
        df_paths : longblob
        density_map : longblob
        density_map_extent : blob
        density_center: blob        
        """
        return definition

    field_table = PlaceholderTable
    swc_table = PlaceholderTable

    def make(self, key):
        swc_path = (self.swc_table() & key).fetch1('swc_path')
        pixel_size_um = (self.field_table() & key).fetch1('pixel_size_um')

        df_paths, density_maps = compute_df_paths_and_density_maps(swc_path=swc_path, pixel_size_um=pixel_size_um)

        # Update cell parameters
        soma_xyz = df_paths[df_paths.type == 1].path[0].flatten()
        density_map = density_maps[1]
        density_map_extent = compute_density_map_extent(paths=df_paths.path.iloc[1:], soma=soma_xyz)
        density_center = compute_density_center(df_paths.path.iloc[1:], soma_xyz, density_map)

        paths_key = key.copy()
        paths_key['soma_xyz'] = np.asarray(soma_xyz).astype(np.float32)
        paths_key['df_paths'] = df_paths.to_dict()
        paths_key['density_map'] = np.asarray(density_map).astype(np.float32)
        paths_key['density_map_extent'] = np.asarray(density_map_extent).astype(np.float32)
        paths_key['density_center'] = np.asarray(density_center).astype(np.float32)

        self.insert1(paths_key)


class LineStackTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Linestack for experiment assumes single cell per exp_num
        -> self.morph_table
        -> self.ipl_table
        ---
        linestack: longblob
        """
        return definition

    morph_table = PlaceholderTable
    ipl_table = PlaceholderTable
    field_table = PlaceholderTable

    def make(self, key):
        zstack_file = (self.field_table() & key).fetch1('fromfile')
        df_paths = pd.DataFrame((self.morph_table() & key).fetch1('df_paths'))
        scaled_ipl_depth = (self.ipl_table() & key).fetch1('scaled_ipl_depth')

        data_stack = load_h5_data(zstack_file, lower_keys=True)
        stack_shape = data_stack.get('line_stack_warped', data_stack['wdatach0']).shape

        linestack = get_linestack(df_paths, stack_shape)

        if scaled_ipl_depth is not None:
            size_diff = scaled_ipl_depth.size - linestack.shape[2]
            if size_diff > 0:
                linestack = np.pad(linestack, ((0, 0), (0, 0), (0, size_diff)), 'constant', constant_values=0)

        paths_key = key.copy()
        paths_key['linestack'] = np.asarray(linestack)

        self.insert1(paths_key)


