import os
import warnings

import datajoint as dj
import numpy as np
import pandas as pd
from roispy3.data_utils import make_dir

from djimaging.tables.morphology.match_utils import get_rec_center_in_stack_coordinates, match_rec_to_stack, \
    create_template, add_soma_to_linestack, calibrate_one_roi, find_roi_pos_stack, compute_roi_pos_metrics, \
    plot_match_template_to_image
from djimaging.tables.morphology.morphology_roi_utils import compute_dendritic_distance_to_soma, \
    plot_roi_positions_xyz
from djimaging.tables.morphology.morphology_utils import get_linestack, compute_df_paths_and_density_maps, \
    compute_density_map_extent, compute_density_center
from djimaging.utils.data_utils import load_h5_data, load_h5_table
from djimaging.utils.datafile_utils import find_folders_with_file_of_type
from djimaging.utils.dj_utils import PlaceholderTable, get_primary_key, make_hash
from djimaging.utils.scanm_utils import get_setup_xscale, extract_roi_idxs


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
        fromfile_flat = '': varchar(255)
        linestack_flat = NULL: longblob
        """
        return definition

    experiment_table = PlaceholderTable
    morph_table = PlaceholderTable
    ipl_table = PlaceholderTable
    field_table = PlaceholderTable

    folder_flat = 'Morph'

    def make(self, key):
        header_path = (self.experiment_table() & key).fetch1('header_path')
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

        flat_file = [f for f in os.listdir(os.path.join(header_path, self.folder_flat)) if f.endswith('flat.tif')]
        assert len(flat_file) <= 1, f'Found multiple flat LineStack files for key={key}'

        paths_key = key.copy()
        paths_key['linestack'] = np.asarray(linestack)

        if len(flat_file) == 1:
            flat_path = os.path.join(header_path, self.folder_flat, flat_file[0])
            import tifffile
            linestack_flat = tifffile.imread(flat_path).T
            paths_key['fromfile_flat'] = flat_path
            paths_key['linestack_flat'] = np.asarray(linestack_flat)

        self.insert1(paths_key)


class RoiStackPosParamsTemplate(dj.Lookup):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        roi_pos_params_hash: varchar(32) # unique param set hash
        ---
        fig_folder='' : varchar(255)
        pad_scale=1.0 : float
        pad_more=50 : int unsigned
        dist_score_factor=1e-3 :float
        soma_radius=10 :float
        match_max_dist_um = 50.0 : float
        match_z_scale = 0.5 : float
        """
        return definition

    def add_default(self, skip_duplicates=False, **update_kw):
        """Add default preprocess parameter to table"""
        key = dict()
        key.update(**update_kw)
        key["roi_pos_params_hash"] = make_hash(key)
        self.insert1(key, skip_duplicates=skip_duplicates)


class FieldStackPosTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.field_table
        -> self.params_table
        ---
        rec_c_warning_flag: tinyint unsigned
        rec_cpos_stack_xy_raw: longblob
        rec_cpos_stack_xyz: longblob
        rec_cpos_stack_fit_dist: float
        """
        return definition

    userinfo_table = PlaceholderTable
    experiment_table = PlaceholderTable
    field_table = PlaceholderTable
    linestack_table = PlaceholderTable
    params_table = PlaceholderTable
    morph_table = PlaceholderTable

    @property
    def key_source(self):
        try:
            return self.field_table.RoiMask() * self.params_table()
        except TypeError:
            pass

    class RoiStackPos(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            -> self.roi_table
            ---
            roi_pos_stack_xyz : blob
            """
            return definition

        roi_table = PlaceholderTable

    class FitInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            ---
            score: float
            score_map : longblob
            linestack_xy : longblob
            template_raw : longblob
            template_fit : longblob
            roi_coords_rot : longblob
            exp_center : blob
            xyz_fit : blob
            rescale=1.0 : float
            """
            return definition

    def make(self, key):
        exp_key = key.copy()
        exp_key.pop('field')

        data_stack_name = (self.userinfo_table() & key).fetch1('data_stack_name')
        rec_filepath, npixartifact = (self.field_table() & key).fetch1('fromfile', 'npixartifact')
        wparams_rec = load_h5_table('wParamsNum', filename=rec_filepath)

        ch_average = create_template(load_h5_data(rec_filepath),
                                     npixartifact=npixartifact, data_stack_name=data_stack_name)

        linestack, stack_filepath = (self.field_table() * self.linestack_table() & exp_key).fetch1(
            'linestack', 'fromfile')
        wparams_stack = load_h5_table('wParamsNum', filename=stack_filepath)

        pixel_size_um_xy, pixel_size_um_z = \
            (self.field_table() * self.field_table.Zstack() * self.linestack_table() & exp_key).fetch1(
                'pixel_size_um', 'zstep')
        pixel_sizes_stack = np.array([pixel_size_um_xy, pixel_size_um_xy, pixel_size_um_z])

        roi_mask = (self.field_table.RoiMask() & key).fetch1('roi_mask')

        setup_xscale = get_setup_xscale((self.experiment_table.ExpInfo() & key).fetch1('setupid'))

        soma_xyz = (self.morph_table() & exp_key).fetch1('soma_xyz')

        params = (self.params_table() & key).fetch1()
        fig_folder = params.pop('fig_folder')
        soma_radius = params.pop('soma_radius')

        if fig_folder is not None:
            make_dir(fig_folder)

            savefilename = fig_folder + '/'
            for k, v in key.items():
                if 'hash' not in k:
                    savefilename += f'{k}_{v}__'

            savefilename = savefilename[:-2] + '.png'

        else:
            savefilename = None

        from IPython.display import clear_output
        clear_output(wait=True)

        rec_cx_stack, rec_cy_stack, rec_c_warning_flag = get_rec_center_in_stack_coordinates(
            wparams_rec=wparams_rec, wparams_stack=wparams_stack,
            linestack=linestack, pixel_sizes_stack=pixel_sizes_stack)

        roi_idxs, rois_pos_stack_xyz, rec_cpos_stack_xyz, fit_dict = match_rec_to_stack(
            ch_average=ch_average, roi_mask=roi_mask, setup_xscale=setup_xscale,
            wparams_rec=wparams_rec, wparams_stack=wparams_stack, pixel_sizes_stack=pixel_sizes_stack,
            linestack=linestack, rec_cxy_stack=np.array([rec_cx_stack, rec_cy_stack]),
            soma_xyz=soma_xyz, soma_radius=soma_radius,
            angle_adjust=0, pad_scale=params['pad_scale'], shift=(0, 0),
            pad_more=params['pad_more'] + 200. if rec_c_warning_flag == 1 else params['pad_more'],
            dist_score_factor=0. if rec_c_warning_flag == 1 else params['dist_score_factor'],
            rescales=None, savefilename=savefilename, seed=42)

        assert set(roi_idxs) == set(np.abs(extract_roi_idxs(roi_mask)))
        assert len(rois_pos_stack_xyz) == len(roi_idxs)

        pos_key = key.copy()
        pos_key['rec_c_warning_flag'] = int(rec_c_warning_flag)
        pos_key['rec_cpos_stack_xy_raw'] = np.array([rec_cx_stack, rec_cy_stack]).astype(np.float32)
        pos_key['rec_cpos_stack_xyz'] = np.asarray(rec_cpos_stack_xyz).astype(np.float32)
        pos_key['rec_cpos_stack_fit_dist'] = \
            np.sum((pos_key['rec_cpos_stack_xy_raw'] - pos_key['rec_cpos_stack_xyz'][:2])**2)**0.5

        info_key = key.copy()
        info_key['score'] = fit_dict['score']
        info_key['score_map'] = fit_dict['score_map']
        info_key['linestack_xy'] = fit_dict['linestack_xy']
        info_key['template_raw'] = fit_dict['template_raw']
        info_key['template_fit'] = fit_dict['template_fit']
        info_key['roi_coords_rot'] = fit_dict['roi_coords_rot']
        info_key['exp_center'] = fit_dict['exp_center']
        info_key['xyz_fit'] = fit_dict['xyz_fit']
        info_key['rescale'] = fit_dict['rescale']

        roi_keys = []
        for roi_idx, roi_pos_stack_xyz in zip(roi_idxs, rois_pos_stack_xyz):
            roi_key = key.copy()
            roi_key['roi_id'] = int(abs(roi_idx))

            artifact_flag = (self.RoiStackPos.roi_table() & roi_key).fetch1('artifact_flag')

            if artifact_flag == 0:
                roi_key['roi_pos_stack_xyz'] = np.asarray(roi_pos_stack_xyz).astype(np.float32)
                roi_keys.append(roi_key)

        self.insert1(pos_key)
        self.FitInfo().insert1(info_key)
        for roi_key in roi_keys:
            self.RoiStackPos().insert1(roi_key)

    def plot1(self, key=None, savefilename=None):
        key = get_primary_key(table=self, key=key)

        fit_dict = (self.FitInfo() & key).fetch1()

        score = fit_dict['score']
        score_map = fit_dict['score_map']
        linestack_xy = fit_dict['linestack_xy']
        xyz_fit = fit_dict['xyz_fit']
        exp_center = fit_dict['exp_center']
        template_fit = fit_dict['template_fit']
        template_raw = fit_dict['template_raw']
        roi_coords_rot = fit_dict['roi_coords_rot']

        plot_match_template_to_image(
            image=linestack_xy, template_fit=template_fit,
            score_map=score_map, best_xy=xyz_fit[:2], best_score=score,
            template_raw=template_raw, roi_coords_rot=roi_coords_rot,
            exp_center=exp_center, savefilename=savefilename)


class FieldCalibratedStackPosTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.fieldstackpos_table
        ---
        success_rate: float
        """
        return definition

    fieldstackpos_table = PlaceholderTable
    linestack_table = PlaceholderTable
    field_table = PlaceholderTable
    morph_table = PlaceholderTable
    params_table = PlaceholderTable

    class RoiCalibratedStackPos(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            -> self.roi_table
            ---
            roi_cal_pos_stack_xyz : blob
            success_cal_flag : tinyint unsigned
            soma_by_dist_flag : tinyint unsigned
            """
            return definition

        roi_table = PlaceholderTable

    def make(self, key):
        exp_key = key.copy()
        exp_key.pop('field')

        linestack, stack_filepath = (self.field_table() * self.linestack_table() & exp_key).fetch1(
            'linestack', 'fromfile')

        pixel_size_um_xy, pixel_size_um_z = \
            (self.field_table() * self.field_table.Zstack() * self.linestack_table() & exp_key).fetch1(
                'pixel_size_um', 'zstep')
        pixel_sizes_stack = np.array([pixel_size_um_xy, pixel_size_um_xy, pixel_size_um_z])

        soma_xyz = (self.morph_table() & exp_key).fetch1('soma_xyz')
        soma_radius, z_scale, max_dist = (self.params_table() & key).fetch1(
            'soma_radius', 'match_z_scale', 'match_max_dist_um')

        soma_linestack = add_soma_to_linestack(
            np.zeros_like(linestack), pixel_sizes_stack,
            soma_xyz, radius_xyz=soma_radius, fill_value=1)

        z_factor = z_scale * pixel_sizes_stack[2] / pixel_sizes_stack[0]

        linestack_coords_xyz = np.vstack(np.where(linestack)).T
        soma_linestack_coords_xyz = np.vstack(np.where(soma_linestack)).T

        soma_stack_xyz = calibrate_one_roi(
            soma_xyz / pixel_sizes_stack, linestack_coords_xyz, z_factor=z_factor, return_dist=False)

        rois_stack_pos = (self.fieldstackpos_table.RoiStackPos() & key).fetch(as_dict=True)

        # Calibrate
        roi_keys = []
        for roi_dict in rois_stack_pos:
            quality, roi_cal_pos_stack_xyz, soma_by_dist = find_roi_pos_stack(
                roi_raw_xyz=roi_dict['roi_pos_stack_xyz'],
                linestack_coords_xyz=linestack_coords_xyz,
                soma_linestack_coords_xyz=soma_linestack_coords_xyz,
                max_dist=max_dist, z_factor=z_factor)

            if soma_by_dist:
                roi_cal_pos_stack_xyz = soma_stack_xyz

            roi_key = key.copy()
            roi_key['roi_id'] = roi_dict['roi_id']
            roi_key['roi_cal_pos_stack_xyz'] = np.asarray(roi_cal_pos_stack_xyz).astype(np.float32)
            roi_key['soma_by_dist_flag'] = abs(int(soma_by_dist))
            roi_key['success_cal_flag'] = int(quality)

            roi_keys.append(roi_key)

        success_rate = np.mean([roi_key['success_cal_flag'] for roi_key in roi_keys])

        # Insert
        field_key = key.copy()
        field_key['success_rate'] = success_rate

        self.insert1(field_key)
        for roi_key in roi_keys:
            self.RoiCalibratedStackPos().insert1(roi_key)


class FieldPosMetricsTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
            -> self.fieldcalibratedstackpos_table
            ---
            max_d_dist: float
            """
        return definition

    fieldcalibratedstackpos_table = PlaceholderTable
    morph_table = PlaceholderTable

    class RoiPosMetrics(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            -> self.roi_table
            ---
            path_id : int
            loc_on_path : blob
            roi_pos_xyz : blob
            roi_pos_xyz_rel_to_soma : blob
            d_dist_to_soma : float
            norm_d_dist_to_soma : float
            ec_dist_to_soma : float
            ec_dist_to_density_center : float
            branches_to_soma : blob
            num_branches_to_soma : int
            """
            return definition

        roi_table = PlaceholderTable

    def make(self, key):
        exp_key = key.copy()
        exp_key.pop('field')

        roi_ids, roi_pos_stack_xyzs = (self.fieldcalibratedstackpos_table.RoiCalibratedStackPos() & key).fetch(
            'roi_id', 'roi_cal_pos_stack_xyz')
        soma_xyz, density_center, df_paths = (self.morph_table() & exp_key).fetch1(
            'soma_xyz', 'density_center', 'df_paths')

        df_paths = pd.DataFrame(df_paths)

        max_d_dist = np.max([compute_dendritic_distance_to_soma(df_paths, i) for i in df_paths.index])

        roi_keys = []
        for roi_id, roi_pos_stack_xyz in zip(roi_ids, roi_pos_stack_xyzs):
            roi_metrics = compute_roi_pos_metrics(roi_pos_stack_xyz, df_paths, soma_xyz, density_center, max_d_dist)

            roi_key = key.copy()
            roi_key['roi_id'] = roi_id
            roi_key.update(**roi_metrics)
            roi_keys.append(roi_key)

        field_key = key.copy()
        field_key['max_d_dist'] = max_d_dist

        self.insert1(field_key)
        for roi_key in roi_keys:
            self.RoiPosMetrics().insert1(roi_key)

    def plot1(self, key=None):
        key = get_primary_key(self, key)

        exp_key = key.copy()
        exp_key.pop('field')

        rois_pos_xyz = np.stack((self.RoiPosMetrics() & exp_key).fetch('roi_pos_xyz')).T
        d_dist_to_soma = (self.RoiPosMetrics() & exp_key).fetch('d_dist_to_soma')
        df_paths = pd.DataFrame((self.morph_table() & exp_key).fetch1('df_paths'))
        soma_xyz = (self.morph_table() & exp_key).fetch1('soma_xyz')
        max_d_dist = np.max((self & exp_key).fetch('max_d_dist'))

        plot_roi_positions_xyz(
            rois_pos_xyz, df_paths.path, soma_xyz, c=d_dist_to_soma, layer_on_z=None, layer_off_z=None,
            roi_max_distance=max_d_dist, plot_rois=True, plot_morph=True, plot_soma=True,
            xlim=None, ylim=None, zlim=None)
