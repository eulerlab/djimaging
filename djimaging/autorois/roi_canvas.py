import os.path
import pickle
import warnings
from typing import Optional, Dict

import numpy as np
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex, hex2color

from djimaging.autorois.autoshift_utils import compute_corr_map, compute_corr_map_match_indexes, extract_best_shift
from djimaging.utils import image_utils, mask_utils, math_utils

try:
    from ipywidgets import HTML, Dropdown, FloatSlider, Button, HBox, VBox, Checkbox, IntSlider, BoundedIntText, \
        IntProgress, FloatProgress, Layout, Output
except ImportError:
    warnings.warn('Failed to import ipywidgets. AutoROIs will not work.')

try:
    from ipycanvas import MultiCanvas, hold_canvas
except ImportError:
    warnings.warn('Failed to import ipycanvas. AutoROIs will not work.')


class RoiCanvas:

    def __init__(
            self,
            ch0_stacks,
            ch1_stacks,
            stim_names=None,
            bg_dict=None,
            shifts=None,
            initial_roi_mask=None,
            main_stim_idx: int = 0,
            n_artifact: int = 0,
            upscale: int = 5,
            autorois_models: Optional[Dict] = None,
            output_files=None,
            pixel_size_um=(1., 1.),
    ):
        """ROI canvas to draw ROIs"""
        for ch0_stack, ch1_stack in zip(ch0_stacks, ch1_stacks):
            if (ch0_stack.shape[:2] != ch0_stacks[0].shape[:2]) or (ch1_stack.shape[:2] != ch0_stacks[0].shape[:2]):
                raise ValueError('Incompatible shapes between stacks')

        if initial_roi_mask is not None and initial_roi_mask.shape != ch0_stacks[0].shape[:2]:
            raise ValueError('Incompatible shapes between ch0_stack and provided roi_mask')

        self.ch0_stacks = ch0_stacks
        self.ch1_stacks = ch1_stacks
        self.other_bgs_dict = bg_dict if bg_dict is not None else dict()

        if shifts is None:
            shifts = np.array([np.array([0, 0], dtype=int) for _ in ch0_stacks])
        self.shifts = np.asarray(shifts)

        if self.shifts.shape[0] != len(self.ch0_stacks):
            raise ValueError('Incompatible shapes between shifts and stacks')

        if stim_names is None:
            stim_names = [f"stim{i + 1}" for i in range(len(self.ch0_stacks))]
        self.stim_names = stim_names
        self.main_stim_idx = main_stim_idx
        self.output_files = output_files

        self.n_artifact = n_artifact
        self.nx = ch0_stacks[0].shape[0]
        self.ny = ch0_stacks[0].shape[1]

        self.upscale = upscale
        self.npx = self.nx * self.upscale
        self.npy = self.ny * self.upscale
        self.pixel_size_um = pixel_size_um

        if autorois_models is not None:
            self.autorois_models = autorois_models
        else:
            self.autorois_models = {'none': None}

        colors = [rgb2hex(rgba[:3]) for rgba in plt.get_cmap('jet')(np.linspace(0, 1, 300))]
        np.random.seed(42)
        np.random.shuffle(colors)
        self.colors = colors

        # Arrays
        self.bg_img = np.zeros((self.nx, self.ny, 4), dtype=int)

        self.current_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.current_mask_img = np.zeros((self.nx, self.ny, 4), dtype=int)

        self.roi_masks = np.zeros((self.nx, self.ny), dtype=int)
        self.roi_mask_img = np.zeros((self.nx, self.ny, 4), dtype=int)

        # Selection
        self._selected_stim_idx = 0
        self._selected_bg = 'ch0_mean'
        self._selected_cmap = 'gray'
        self._selected_gamma = 0.33
        self._selected_roi = 1
        self._selected_view = 'highlight'
        self._selected_tool = 'draw' if initial_roi_mask is None else 'select'
        self._selected_size = 1
        self._selected_thresh = 0.3

        self.alpha_bg = 200. / 255
        self.alpha_hl = 150. / 255
        self.alpha_ft = 50. / 255

        self.backgrounds = self.compute_backgrounds()
        self.output_file = self.output_files[self._selected_stim_idx]

        self.ref_corr_map = None

        # Initialize roi_mask
        if initial_roi_mask is not None:
            self._initial_roi_mask = initial_roi_mask.copy()
            self.init_roi_mask(self._initial_roi_mask)
        else:
            self._initial_roi_mask = None

    def __repr__(self):
        return f"RoiCanvas({self.nx}x{self.ny})"

    def compute_backgrounds(self):
        shift = self.shifts[self._selected_stim_idx]
        shift = np.array([shift[0], shift[1]])
        backgrounds = {
            'ch0_mean': mask_utils.shift_array(np.mean(self.ch0_stacks[self._selected_stim_idx], axis=2), shift),
            'ch0_std': mask_utils.shift_array(np.std(self.ch0_stacks[self._selected_stim_idx], axis=2), shift),
            'ch1_mean': mask_utils.shift_array(np.mean(self.ch1_stacks[self._selected_stim_idx], axis=2), shift),
            'ch1_std': mask_utils.shift_array(np.std(self.ch1_stacks[self._selected_stim_idx], axis=2), shift),
            'none': np.full((self.nx, self.ny), 255)
        }

        if self.other_bgs_dict is not None:
            for key, bg in self.other_bgs_dict.items():
                backgrounds[key] = bg

        return backgrounds

    def update_backgrounds(self):
        self.backgrounds = self.compute_backgrounds()

    def reset_current_mask(self):
        """Set current mask and the respective image to zero"""
        self.current_mask[:] = False
        self.current_mask_img[:] = 0

    def reset_roi_masks(self):
        """Set all masks and the respective image to zero"""
        self.roi_masks[:] = 0
        self.roi_mask_img[:] = 0

    def init_roi_mask(self, roi_mask):
        self.reset_roi_masks()
        for roi_idx in np.unique(roi_mask[roi_mask > 0]):
            self._selected_roi = roi_idx
            self.reset_current_mask()
            self.add_to_current_mask(np.asarray(roi_mask == roi_idx))
            self.add_current_mask_to_roi_masks()
        self.roi_masks = mask_utils.relabel_mask(self.roi_masks, connectivity=2)

    def add_to_current_mask(self, mask):
        """Add a mask to the current mask"""
        self.current_mask |= mask.astype(bool)

    def load_current_mask(self):
        """Load current mask from all masks"""
        self.reset_current_mask()
        self.current_mask[self.roi_masks == self._selected_roi] = self.roi_masks[self.roi_masks == self._selected_roi]

    def _get_roi_rgb255(self, roi):
        """Get color for ROI"""
        return (255 * np.array(hex2color(self.colors[roi % len(self.colors)]))).astype(int)

    def _get_roi_alpha255(self, roi):
        """Get alpha for ROI"""
        if self._selected_view == 'all' or (
                self._selected_view in ['highlight', 'selected'] and roi == self._selected_roi):
            return int(self.alpha_hl * 255)
        elif self._selected_view == 'selected' and roi != self._selected_roi:
            return 0
        else:
            return int(self.alpha_ft * 255)

    def _get_roi_rgba255(self, roi):
        """Get color and alpha for ROI"""
        rgb = self._get_roi_rgb255(roi)
        a = self._get_roi_alpha255(roi)
        return np.append(rgb, a)

    def update_current_mask_img(self):
        """Update current mask image based on current mask"""
        self.current_mask_img[:] = 0
        if self._selected_view == 'none':
            return
        ys, xs = np.where(self.current_mask)
        self.current_mask_img[ys, xs, :] = self._get_roi_rgba255(self._selected_roi)

    def update_bg_img(self):
        self.bg_img = image_utils.color_image(
            self.backgrounds[self._selected_bg],
            cmap=self._selected_cmap, gamma=self._selected_gamma, alpha=int(self.alpha_bg * 255))

    def erase_from_masks(self, x, y):
        """Remove pixels from current mask and all masks"""
        mask = mask_utils.create_circular_mask(w=self.nx, h=self.ny, center=(x, y), radius=self._selected_size - 1)
        self.current_mask[mask] = 0
        self.roi_masks[mask] = 0  # TODO: allow to undo erase

    def add_current_mask_to_roi_masks(self):
        """Add the current mask to all masks and reset current mask"""
        self.roi_masks[self.current_mask] = self._selected_roi
        self.reset_current_mask()

    def get_selected_stack(self):
        """Get selected stack"""
        return self.ch0_stacks[self._selected_stim_idx]

    def get_ref_stack(self):
        """Get reference stack"""
        return self.ch0_stacks[self.main_stim_idx]

    def compute_autoshift(self, shift_max=5, fun_progress=None):
        self.ref_corr_map = self.ref_corr_map if self.ref_corr_map is not None else compute_corr_map(
            self.get_ref_stack(), fun_progress=fun_progress)
        corr_map = compute_corr_map(self.get_selected_stack(), fun_progress=fun_progress)
        match_indexes = compute_corr_map_match_indexes(
            corr_map, self.ref_corr_map, shift_max=shift_max, fun_progress=fun_progress)
        shift_x, shift_y = extract_best_shift(match_indexes)
        return -shift_x, -shift_y  # Flip sign to get from field to roi mask shift

    def update_roi_mask_img(self):
        """Update ROI masks image"""
        self.roi_mask_img = np.zeros_like(self.roi_mask_img)

        if self._selected_view == 'none':
            return

        new_roi_mask_img = np.zeros_like(self.roi_mask_img)

        for roi in np.unique(self.roi_masks)[1:]:
            xs, ys = np.where(self.roi_masks == roi)
            new_roi_mask_img[xs, ys, :] = self._get_roi_rgba255(roi)

        self.roi_mask_img = new_roi_mask_img

    def print_state(self):
        print('bg'.ljust(15), self._selected_bg)
        print('roi'.ljust(15), self._selected_roi)
        print('view'.ljust(15), self._selected_view)
        print('tool'.ljust(15), self._selected_tool)
        print('size'.ljust(15), self._selected_size)
        print('cc'.ljust(15), self._selected_thresh)

    def plot_layers(self):
        """Plot all data layers. For debugging mostly."""
        fig, axs = plt.subplots(4, 1, figsize=(8, 8))

        axs[0].set_title('current_mask')
        _current_mask = self.current_mask.copy().astype(float)
        _current_mask[_current_mask == 0] = np.nan
        im = axs[0].imshow(_current_mask, vmin=-0.5, vmax=9.5, cmap='tab10')
        plt.colorbar(im, ax=axs[0])

        axs[1].set_title('current_mask_img')
        axs[1].imshow(self.current_mask_img)

        axs[2].set_title('roi_masks')
        _roi_masks = self.roi_masks.copy().astype(float)
        _roi_masks[_roi_masks == 0] = np.nan
        im = axs[2].imshow(_roi_masks, vmin=-0.5, vmax=9.5, cmap='tab10')
        plt.colorbar(im, ax=axs[2])

        axs[3].set_title('roi_mask_img')
        axs[3].imshow(self.roi_mask_img)

        plt.tight_layout()
        plt.show()


class InteractiveRoiCanvas(RoiCanvas):

    def __init__(
            self,
            ch0_stacks,
            ch1_stacks,
            stim_names=None,
            bg_dict=None,
            main_stim_idx=0,
            initial_roi_mask=None,
            shifts=None,
            n_artifact=0,
            upscale=5,
            canvas_width=50,
            autorois_models: Optional[Dict] = None,
            output_files=None,
            pixel_size_um=(1., 1.),
            show_diagnostics=False,
    ):
        super().__init__(ch0_stacks=ch0_stacks, ch1_stacks=ch1_stacks, stim_names=stim_names,
                         bg_dict=bg_dict, n_artifact=n_artifact,
                         initial_roi_mask=initial_roi_mask, shifts=shifts, main_stim_idx=main_stim_idx,
                         upscale=upscale, autorois_models=autorois_models,
                         output_files=output_files, pixel_size_um=pixel_size_um)
        # Create menu elements
        self.current_traces = np.array([])
        self.current_cc = np.array([])
        self.widget_progress = self.create_widget_progress()

        self.widget_diagnostics = self.create_widget_diagnostics() if show_diagnostics else None

        self.widget_bg = self.create_widget_bg()
        self.widget_cmap = self.create_widget_cmap()

        self.widget_roi = self.create_widget_roi()
        self.widget_roi_next = self.create_widget_roi_next()
        self.widget_roi_prev = self.create_widget_roi_prev()

        self.widget_view = self.create_widget_view()
        self.widget_alpha_bg = self.create_widget_alpha_bg()
        self.widget_alpha_hl = self.create_widget_alpha_hl()
        self.widget_alpha_ft = self.create_widget_alpha_ft()

        self.widget_gamma = self.create_widget_gamma()
        self.widget_tool = self.create_widget_tool()
        self.widget_size = self.create_widget_size()
        self.widget_thresh = self.create_widget_thresh()
        self.widget_shift_dx = self.create_widget_shift(axis='x')
        self.widget_shift_dy = self.create_widget_shift(axis='y')
        self.widget_auto_shift = self.create_widget_auto_shift()

        self.widget_sel_stim = self.create_widget_sel_stim()

        self.widget_sel_autorois = self.create_widget_sel_autorois()
        self.widget_exec_autorois = self.create_widget_exec_autorois()
        self.widget_exec_autorois_missing = self.create_widget_exec_autorois_missing()

        self.widget_save_and_new = self.create_widget_save_and_new()
        self.widget_undo = self.create_widget_undo()
        self.widget_save = self.create_widget_save()
        self.widget_info = self.create_widget_info()
        self.widget_clean = self.create_widget_clean()
        self.widget_kill_roi = self.create_widget_kill_roi()
        self.widget_kill_all = self.create_widget_kill_all()
        self.widget_reset_all = self.create_widget_reset_all()
        self.widget_save_to_file = self.create_widget_save_to_file()
        self.widget_save_all_to_file = self.create_widget_save_all_to_file()
        self.widget_save_info = self.create_widget_save_info()

        self.widget_read_only = self.create_widget_read_only()
        self.widget_dangerzone = self.create_widget_dangerzone()

        # ToDo: Add AutoROIs on one or all stimuli widget
        # ToDo: Add AutoROIs to keep existing ROI masks.

        # Create canvas
        self.img = MultiCanvas(n_canvases=4, width=self.npx, height=self.npy)
        self.img.layout.width = f'{canvas_width}%'
        self.img.layout.height = 'auto'
        self.img.sync_image_data = True

        self.canvas_k = self.img[0]
        self.canvas_k.fill_rect(x=0, y=0, width=self.npx, height=self.npy)
        self.canvas_bg = self.img[1]
        self.canvas_masks = self.img[2]
        self.canvas = self.img[3]

        self.drawing = False
        self.last_mouse_pos_xy = (-1, -1)

        self.canvas.on_mouse_down(self.on_mouse_down)
        self.canvas.on_mouse_move(self.on_mouse_move)
        self.canvas.on_mouse_up(self.on_mouse_up)

        if self._initial_roi_mask is not None:
            self.set_read_only(True)

        self.update_all()
        self.draw_all(update=True)

    def start_gui(self):
        data_lines = [self.img, self.widget_diagnostics] if self.widget_diagnostics is not None else [self.img]
        control_lines = [
            self.widget_progress,
            HBox((self.widget_read_only, self.widget_dangerzone,)),
            self.widget_sel_stim,
            HBox((self.widget_roi, self.widget_roi_prev, self.widget_roi_next)),
            HBox((self.widget_save_and_new, self.widget_save, self.widget_undo)),
            HBox((self.widget_tool, self.widget_size, self.widget_thresh)),
            HBox((self.widget_view, self.widget_alpha_bg, self.widget_alpha_hl, self.widget_alpha_ft)),
            HBox((self.widget_bg, self.widget_gamma, self.widget_cmap)),
            HBox((self.widget_shift_dx, self.widget_shift_dy, self.widget_auto_shift)),
            HBox(
                (self.widget_sel_autorois, self.widget_exec_autorois, self.widget_exec_autorois_missing)),
            self.widget_info,
            HBox((self.widget_clean, self.widget_kill_roi, self.widget_kill_all, self.widget_reset_all)),
            HBox((self.widget_save_to_file, self.widget_save_all_to_file, self.widget_save_info)),
        ]

        # concatenate the two line tuple
        vbox = VBox(data_lines + control_lines)
        return vbox

    def set_selected_cmap(self, value):
        self.widget_cmap.value = value
        self._selected_cmap = value
        self.draw_bg()

    def create_widget_cmap(self):
        """Create and return button"""
        options = ['Greys_r', 'bone', 'gray', 'viridis', 'jet', 'plasma', 'inferno', 'magma', 'cividis']
        widget = Dropdown(options=options, value=self._selected_cmap, description='Cmap:', disabled=False)

        def change(value):
            self.set_selected_cmap(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_bg(self, value):
        self.widget_bg.value = value
        self._selected_bg = value
        self.draw_bg()

    def create_widget_bg(self):
        """Create and return button"""
        options = list(self.backgrounds.keys())
        widget = Dropdown(options=options, value=self._selected_bg, description='BG:', disabled=False)

        def change(value):
            self.set_selected_bg(value['new'])

        widget.observe(change, names='value')
        return widget

    def create_widget_alpha_bg(self):
        widget = FloatSlider(min=0, max=1, step=0.1, value=self.alpha_bg, description='α[BG]:',
                             disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                             readout_format='.2f', layout=Layout(width='250px'))

        def change(value):
            self.set_selected_alpha_bg(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_alpha_bg(self, value):
        self.widget_alpha_bg.value = value
        self.alpha_bg = value
        self.draw_bg(update=True)

    def create_widget_alpha_hl(self):
        widget = FloatSlider(min=0, max=1, step=0.01, value=self.alpha_hl, description='α[HL]:',
                             disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                             readout_format='.2f', layout=Layout(width='250px'))

        def change(value):
            self.set_selected_alpha_hl(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_alpha_hl(self, value):
        self.widget_alpha_hl.value = value
        self.alpha_hl = value
        self.draw_current_mask_img(update=True)

    def create_widget_alpha_ft(self):
        widget = FloatSlider(min=0, max=1, step=0.01, value=self.alpha_ft, description='α[FA]:',
                             disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                             readout_format='.2f', layout=Layout(width='250px'))

        def change(value):
            self.set_selected_alpha_ft(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_alpha_ft(self, value):
        self.widget_alpha_ft.value = value
        self.alpha_ft = value
        self.draw_roi_masks_img(update=True)

    def create_widget_dangerzone(self):
        widget = Checkbox(value=False, description='Dangerzone', disabled=False)

        def change(state):
            self.set_dangerzone(state['new'])

        widget.observe(change, names='value')
        return widget

    def set_dangerzone(self, dangerzone):
        self.widget_dangerzone.value = dangerzone
        self.widget_kill_all.disabled = not dangerzone
        self.widget_reset_all.disabled = not dangerzone

    def create_widget_read_only(self):
        widget = Checkbox(value=False, description='Read only', disabled=False)

        def change(state):
            self.set_read_only(state['new'])

        widget.observe(change, names='value')
        return widget

    def set_read_only(self, read_only, force=False):
        self.widget_read_only.value = read_only
        self.widget_save.disabled = read_only
        self.widget_save_and_new.disabled = read_only
        self.widget_undo.disabled = read_only
        self.widget_dangerzone.disabled = read_only
        self.widget_kill_roi.disabled = read_only
        self.widget_tool.disabled = read_only
        self.widget_clean.disabled = read_only
        self.widget_exec_autorois.disabled = read_only
        self.widget_exec_autorois_missing.disabled = read_only

        if read_only:
            self.set_selected_tool('select')
            self.set_dangerzone(False)

            if force:
                self.widget_read_only.disabled = True

        self.update_shift_widget()

    def create_widget_progress(self):
        widget = FloatProgress(
            value=100,
            min=0,
            max=100,
            description='Progress:',
            bar_style='success',
            orientation='horizontal'
        )
        return widget

    def create_widget_sel_stim(self):
        widget = Dropdown(options=self.stim_names, value=self.stim_names[self._selected_stim_idx],
                          description='Stimulus:', disabled=False)

        def change(value):
            self.set_selected_stim(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_stim(self, value):
        self.widget_sel_stim.value = value
        self._selected_stim_idx = np.argmax([stim_name == value for stim_name in self.stim_names])
        self.output_file = self.output_files[self._selected_stim_idx]
        self.widget_save_info.value = f'{self.output_file}'
        self.update_backgrounds()
        self.update_shift_widget()
        self.update_info()
        self.draw_bg(update=True)

    def update_shift_widget(self):
        if self._selected_stim_idx == self.main_stim_idx:
            self.widget_shift_dx.disabled = True
            self.widget_shift_dy.disabled = True
            self.widget_auto_shift.disabled = True
        else:
            self.widget_shift_dx.disabled = self.widget_read_only.value
            self.widget_shift_dy.disabled = self.widget_read_only.value
            self.widget_auto_shift.disabled = self.widget_read_only.value

        self.widget_shift_dx.value, self.widget_shift_dy.value = self.shifts[self._selected_stim_idx]

    def create_widget_sel_autorois(self):
        """Create and return button"""
        options = [key for key in self.autorois_models]
        widget = Dropdown(options=options, value=options[0], description='AutoROIs:', disabled=False)

        def change(value):
            new_value = value["new"]
            self.widget_sel_autorois.value = new_value

        widget.observe(change, names='value')
        return widget

    def create_widget_exec_autorois(self):
        widget_exec_autorois = Button(description='Detect ROIs', disabled=False, button_style='success')
        widget_exec_autorois.on_click(self.exec_autorois_all)
        return widget_exec_autorois

    def exec_autorois_all(self, button=None):
        self._exec_autorois(missing_only=False)

    def _exec_autorois(self, missing_only=False):
        current_model = self.widget_sel_autorois.value
        model = self.autorois_models[current_model]
        if model is not None:
            self.update_progress(0)
            self.set_read_only(True, force=True)

            roi_mask_new = model.create_mask_from_data(
                ch0_stack=self.ch0_stacks[self._selected_stim_idx],
                ch1_stack=self.ch1_stacks[self._selected_stim_idx],
                n_artifact=self.n_artifact,
                pixel_size_um=self.pixel_size_um,
            )

            if missing_only:
                roi_mask_old = self.roi_masks.copy()
                roi_mask_new, new_labels = mask_utils.add_rois(roi_mask_new, roi_mask_old, connectivity=2, plot=False)

            self.update_progress(50)

            self.init_roi_mask(roi_mask_new)
            self.update_info()
            self.update_roi_options()
            self.draw_current_mask_img(update=True)
            self.draw_roi_masks_img(update=True)
            self.set_dangerzone(False)
            self.set_selected_tool('select')

            self.update_progress(100)
            self.set_read_only(False)

    def create_widget_exec_autorois_missing(self):
        widget = Button(description='Add missing ROIs', disabled=False, button_style='success')
        widget.on_click(self.exec_autorois_missing)
        return widget

    def exec_autorois_missing(self, button=None):
        self._exec_autorois(missing_only=True)

    def set_new_roi_mask(self, roi_mask):
        self.init_roi_mask(roi_mask)
        self.update_info()
        self.update_roi_options()
        self.draw_current_mask_img(update=True)
        self.draw_roi_masks_img(update=True)

    def set_selected_roi(self, value):
        if value == self._selected_roi:
            return

        self._selected_roi = value
        self.widget_roi.value = value
        self.load_current_mask()
        self.update_info()
        self.draw_current_mask_img(update=True)
        self.draw_roi_masks_img(update=True)

    def create_widget_roi(self):
        """Create and return button"""
        options = np.unique(np.append(self.roi_masks[self.roi_masks > 0].flatten(), self._selected_roi))

        widget = Dropdown(options=options, value=self._selected_roi, description='ROI:', disabled=False)

        def change(value):
            self.set_selected_roi(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_view(self, value):  # TODO: Add border view
        self._selected_view = value
        self.draw_current_mask_img(update=True)
        self.draw_roi_masks_img(update=True)

    def create_widget_view(self):
        """Create and return button"""
        options = ['highlight', 'all', 'selected', 'faint', 'none']
        widget = Dropdown(options=options, value=self._selected_view, description='View:', disabled=False)

        def change(value):
            self.set_selected_view(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_tool(self, value):
        self._selected_tool = value
        if self._selected_tool == 'select':
            self.widget_thresh.disabled = True
            self.widget_size.disabled = True
        elif self._selected_tool == 'bg':
            self.widget_thresh.disabled = False
            self.widget_size.disabled = False
            self.set_selected_thresh(0.9)
        elif self._selected_tool == 'cc':
            self.widget_thresh.disabled = False
            self.widget_size.disabled = False
            self.set_selected_thresh(0.3)
        else:
            self.widget_thresh.disabled = True
            self.widget_size.disabled = False

    def create_widget_tool(self):
        """Create and return button"""
        options = ['draw', 'erase', 'cc', 'select', 'bg']
        widget = Dropdown(options=options, value=self._selected_tool, description='Tool:', disabled=False)

        def change(value):
            self.set_selected_tool(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_size(self, value):
        self._selected_size = value

    def create_widget_size(self):
        """Create and return button"""
        widget = FloatSlider(min=1, max=10.0, step=0.5, value=self._selected_size,
                             description='Size:', disabled=False, continuous_update=False, orientation='horizontal',
                             readout=True, readout_format='.1f')

        def change(value):
            self.set_selected_size(value['new'])

        widget.observe(change, names='value')
        return widget

    def get_shift(self, axis):
        if axis == 'x':
            return self.shifts[self._selected_stim_idx, 0]
        else:
            return self.shifts[self._selected_stim_idx, 1]

    def set_shift(self, value, axis, update=True):
        if axis == 'x':
            self.widget_shift_dx.value = value
            self.shifts[self._selected_stim_idx, 0] = value
        else:
            self.widget_shift_dy.value = value
            self.shifts[self._selected_stim_idx, 1] = value

        self.widget_exec_autorois.disabled = self.widget_shift_dx.value != 0 or self.widget_shift_dy.value != 0
        self.widget_exec_autorois_missing.disabled = self.widget_shift_dx.value != 0 or self.widget_shift_dy.value != 0

        if update:
            self.update_backgrounds()
            self.update_info()
            self.draw_bg(update=True)

    def create_widget_shift(self, axis):
        value = self.get_shift(axis=axis)

        widget = BoundedIntText(
            value=value, min=-5, max=5, step=1,
            description=f'shift_d{axis}:', disabled=False,
            continuous_update=False,
            layout=Layout(width='150px', description_width='50px'),
        )

        def change(value):
            self.set_shift(value['new'], axis=axis)

        widget.observe(change, names='value')
        return widget

    def create_widget_auto_shift(self):
        widget = Button(description='Auto shift', disabled=False, button_style='success')
        widget.on_click(self.exec_auto_shift)
        return widget

    def update_progress(self, percent):
        assert 0 <= percent <= 100
        self.widget_progress.value = percent

    def exec_auto_shift(self, button=None):
        self.set_read_only(True, force=True)
        shift_x, shift_y = self.compute_autoshift(fun_progress=self.update_progress)
        self.set_shift(value=shift_x, axis='x', update=False)
        self.set_shift(value=shift_y, axis='y', update=True)
        self.set_read_only(False)

    def set_selected_thresh(self, value):
        self._selected_thresh = value

    def create_widget_thresh(self):
        """Create and return button"""
        widget = FloatSlider(min=0.0, max=1.0, step=0.01, value=self._selected_thresh,
                             description='Thresh:', disabled=self._selected_tool not in ['cc', 'bg'],
                             continuous_update=False, orientation='horizontal',
                             readout=True, readout_format='.2f')

        def change(value):
            self.set_selected_thresh(value['new'])

        widget.observe(change, names='value')
        return widget

    def set_selected_gamma(self, value):
        self._selected_gamma = value
        self.draw_bg(update=True)

    def create_widget_gamma(self):
        """Create and return button"""
        widget = FloatSlider(min=0.1, max=3.0, step=0.1, value=self._selected_gamma,
                             description='Gamma:',
                             continuous_update=False, orientation='horizontal',
                             readout=True, readout_format='.1f')

        def change(value):
            self.set_selected_gamma(value['new'])

        widget.observe(change, names='value')
        return widget

    def exec_undo(self, button=None):
        self.reset_current_mask()
        self.draw_current_mask_img(update=True)
        self.update_info()

    def create_widget_undo(self):
        widget_undo = Button(description='Undo', disabled=False, button_style='warning')
        widget_undo.on_click(self.exec_undo)
        return widget_undo

    def exec_save(self, button=None):
        self.add_current_mask_to_roi_masks()
        self.draw_current_mask_img(update=True)
        self.draw_roi_masks_img(update=True)

    def create_widget_save(self):
        widget_save = Button(description='Save', disabled=False, button_style='success')
        widget_save.on_click(self.exec_save)
        return widget_save

    def exec_save_and_new(self, button=None):
        self.add_current_mask_to_roi_masks()
        self.update_roi_options()
        self.set_selected_roi(self.widget_roi.options[-1])
        self.load_current_mask()
        self.draw_current_mask_img(update=True)
        self.draw_roi_masks_img(update=True)

    def create_widget_save_and_new(self):
        widget_save_and_new = Button(description='New ROI', disabled=False, button_style='success')
        widget_save_and_new.on_click(self.exec_save_and_new)
        return widget_save_and_new

    def exec_clean(self, button=None):
        self.set_read_only(True, force=True)
        self.exec_save_and_new()
        self.set_selected_roi(value=1)
        self.roi_masks = mask_utils.clean_rois(
            self.roi_masks, self.n_artifact, min_size=3, connectivity=2, verbose=False)
        self.draw_all(update=True)
        self.set_read_only(False)

    def create_widget_clean(self):
        widget = Button(description='Clean mask', disabled=False, button_style='warning')
        widget.on_click(self.exec_clean)
        return widget

    def exec_kill_roi(self, button=None):
        self.reset_current_mask()
        self.roi_masks[self.roi_masks == self._selected_roi] = 0
        self.draw_current_mask_img(update=True)
        self.draw_roi_masks_img(update=True)

    def create_widget_kill_roi(self):
        widget_kill_roi = Button(description='Kill selected ROI', disabled=False, button_style='warning')
        widget_kill_roi.on_click(self.exec_kill_roi)
        return widget_kill_roi

    def exec_kill_all_rois(self, button=None):
        self.reset_roi_masks()
        self.update_roi_options()
        self.set_dangerzone(False)

    def create_widget_kill_all(self):
        widget_kill_all = Button(description='Kill ALL ROIs', disabled=True, button_style='danger')
        widget_kill_all.on_click(self.exec_kill_all_rois)
        return widget_kill_all

    def create_widget_reset_all(self):
        widget = Button(description='Reset ALL ROIs', disabled=True, button_style='danger')
        widget.on_click(self.exec_reset_all_rois)
        return widget

    def exec_reset_all_rois(self, button=None):
        if self._initial_roi_mask is not None:
            self.set_new_roi_mask(self._initial_roi_mask)
            self.set_dangerzone(False)

    def create_widget_save_to_file(self):
        widget = Button(description='Save to file', disabled=False, button_style='success')
        widget.on_click(self.exec_save_to_file)
        return widget

    def prep_roi_mask_for_file(self):
        roi_mask = self.roi_masks.copy()
        roi_shift = self.shifts[self._selected_stim_idx]
        roi_shift = np.array([-roi_shift[0], -roi_shift[1]])
        roi_mask = mask_utils.shift_array(img=roi_mask, shift=roi_shift, inplace=True, cval=0)
        return roi_mask

    def exec_save_to_file(self, botton=None):
        roi_mask = self.prep_roi_mask_for_file()

        with open(self.output_file, 'wb') as f:
            pickle.dump(roi_mask, f)

        success = os.path.isfile(self.output_file)
        if success:
            self.widget_save_info.description = 'Success!'.ljust(20)
        else:
            self.widget_save_info.description = 'Error!'.ljust(20)

    def create_widget_save_all_to_file(self):
        widget = Button(description='Save all to file', disabled=False, button_style='success')
        widget.on_click(self.exec_save_all_to_file)
        return widget

    def exec_save_all_to_file(self, button=None):
        import time
        self.update_progress(0)
        for i, stim in enumerate(self.stim_names, start=1):
            self.set_selected_stim(stim)
            try:
                self.exec_save_to_file()
            except (OSError, FileNotFoundError) as e:
                self.widget_save_info.description = 'Error!'.ljust(20)
                warnings.warn(f'Error saving {stim}: {e}')
            time.sleep(0.5)
            self.update_progress(100 * i / len(self.stim_names))

    def create_widget_save_info(self):
        widget = HTML(value=f'{self.output_file}', placeholder='output_file', description=''.ljust(20))
        return widget

    def get_current_stack(self):
        current_stack = mask_utils.shift_array(
            self.ch0_stacks[self._selected_stim_idx], (self.get_shift('x'), self.get_shift('y')))
        return current_stack

    def update_info(self):
        self.current_traces = self._compute_traces()
        self.current_cc = self._compute_cc(self.current_traces)

        self.update_diagnostics(update=False)
        self.update_cc()

    def _compute_traces(self):
        return self.get_current_stack()[self.current_mask]

    def _compute_cc(self, traces):
        return np.corrcoef(traces)

    def update_diagnostics(self, update=True):
        if self.widget_diagnostics is None:
            return

        with self.widget_diagnostics:
            clear_output(wait=True)
            fig, axs = plt.subplots(1, 3, figsize=(7, 1.5), width_ratios=(2, 1, 1))
            if np.any(self.current_mask):
                traces = self._compute_traces() if update else self.current_traces

                axs[0].plot(traces.T, c='dimgray', lw=0.2)
                axs[0].plot(np.mean(traces, axis=0), c='k', lw=1)

                if np.sum(self.current_mask) > 1:
                    cc = self._compute_cc(traces) if update else self.current_cc
                    axs[1].hist(cc[np.triu_indices_from(cc, k=1)])
                    vabsmax = np.maximum(0.1, np.max(np.abs(cc[np.triu_indices_from(cc, k=1)])))
                    axs[1].set(xlim=(-vabsmax * 1.05, vabsmax * 1.05))

                    mask = self.current_mask
                    xs, ys = np.where(mask)

                    # map cc values onto mask
                    cc_mask = np.full(mask.shape, np.nan, dtype=np.float32)

                    cc = cc.copy()
                    cc[np.diag_indices_from(cc)] = np.nan
                    cc_mask[xs, ys] = np.nanmean(cc, axis=0)

                    vabsmax = np.maximum(0.1, np.nanmax(np.abs(cc_mask)))
                    im = axs[2].imshow(cc_mask.T, origin='lower', cmap='coolwarm', vmin=-vabsmax, vmax=vabsmax)
                    plt.colorbar(im, ax=axs[2])
                    axs[2].set(xlim=(np.min(xs) - 3, np.max(xs) + 3), ylim=(np.min(ys) - 3, np.max(ys) + 3))

                plt.tight_layout()
            else:
                for ax in axs.flat:
                    ax.axis('off')

            plt.show()

    def update_cc(self, update=True):
        n_px = np.sum(self.current_mask)
        if np.sum(self.current_mask) >= 2:
            cc = self._compute_cc(self._compute_traces()) if update else self.current_cc
            cc = cc[np.triu_indices_from(cc, k=1)]
            self.widget_info.value = f"n_px={n_px}, min_cc={np.min(cc):.2f}, mean_cc={np.mean(cc):.2f}"
        else:
            self.widget_info.value = f"n_px={n_px}"

    @staticmethod
    def create_widget_info():
        widget_info = HTML(value="--", placeholder='ROI cc:', description='ROI cc')
        return widget_info

    def update_roi_options(self):
        max_roi = np.max(np.unique(self.roi_masks)) + 1
        self.widget_roi.disabled = True
        self.widget_roi.options = tuple(np.unique(self.roi_masks)[1:]) + (max_roi,)
        new_roi = self._selected_roi if self._selected_roi in self.widget_roi.options else self.widget_roi.options[0]
        self.widget_roi.disabled = False
        self.set_selected_roi(new_roi)

    def _apply_tool(self, x, y, down):
        """Apply tool. Down means mouse down event"""

        ix = int(x / self.upscale)
        iy = self.ny - int(y / self.upscale) - 1  # Flip ud

        print(f'Apply tool={self._selected_tool} at x={x}, y={y}, ix={ix}, iy={iy}')

        if self._selected_tool == 'draw':
            mask = mask_utils.create_circular_mask(
                w=self.nx, h=self.ny, center=(ix, iy), radius=self._selected_size - 1)
            self.add_to_current_mask(mask=mask)
        elif self._selected_tool == 'erase':
            self.erase_from_masks(x=ix, y=iy)
        elif self._selected_tool == 'cc':
            mask = mask_utils.get_mask_by_cc(
                seed_ix=ix, seed_iy=iy, data=self.ch0_stacks[self._selected_stim_idx],
                plot=False, thresh=self._selected_thresh, max_pixel_dist=self._selected_size)
            self.add_to_current_mask(mask=mask)
        elif self._selected_tool == 'bg':
            img = math_utils.normalize_zero_one(np.mean(self.bg_img, axis=2))
            if self.bg_img.shape[0] != self.nx or self.bg_img.shape[1] != self.ny:
                img = image_utils.resize_image(img, output_shape=(self.nx, self.ny))
            mask = mask_utils.get_mask_by_bg(
                seed_ix=ix, seed_iy=iy,
                data=img, plot=False, thresh=1. - self._selected_thresh,
                max_pixel_dist=self._selected_size)
            self.add_to_current_mask(mask=mask)
        elif self._selected_tool == 'select' and down:
            if self.roi_masks[ix, iy] > 0:
                self.set_selected_roi(self.roi_masks[ix, iy])

    def on_mouse_down(self, x, y):
        """React to mouse done on canvas, e.g. by drawing on canvas"""
        self.drawing = True

        self._apply_tool(x, y, down=True)
        self.last_mouse_pos_xy = (int(x / self.upscale), int(y / self.upscale))

        self.draw_current_mask_img(update=True)
        if self._selected_tool == 'erase':
            self.draw_roi_masks_img(update=True)

    def on_mouse_move(self, x, y):
        if not self.drawing:
            return

        if self.last_mouse_pos_xy == (int(x / self.upscale), int(y / self.upscale)):
            return

        self.last_mouse_pos_xy = (int(x / self.upscale), int(y / self.upscale))
        self._apply_tool(x, y, down=False)

        self.draw_current_mask_img(update=True)
        if self._selected_tool == 'erase':
            self.draw_roi_masks_img(update=True)

    def on_mouse_up(self, x, y):
        self.last_mouse_pos_xy = (-1, -1)
        self.drawing = False
        if self._selected_tool != 'select':
            self.update_info()
        self.draw_roi_masks_img()

    # Draw functions
    def draw_current_mask_img(self, update=True):

        if update:
            self.update_current_mask_img()

        img = image_utils.upscale_image(self.current_mask_img, upscale=self.upscale)
        img = np.flipud(np.swapaxes(img, 0, 1))

        with hold_canvas(self.canvas):
            self.canvas.clear()
            self.canvas.put_image_data(img, 0, 0)

    def draw_roi_masks_img(self, update=True):

        if update:
            self.update_roi_mask_img()

        img = image_utils.upscale_image(self.roi_mask_img, upscale=self.upscale)
        img = np.flipud(np.swapaxes(img, 0, 1))

        with hold_canvas(self.canvas_masks):
            self.canvas_masks.clear()
            self.canvas_masks.put_image_data(img, 0, 0)

    def draw_bg(self, update=True):

        if update:
            self.update_bg_img()

        if self.bg_img.shape[:2] == (self.nx, self.ny):
            print('Upscaling background image')
            img = image_utils.upscale_image(self.bg_img, self.upscale)
        elif self.bg_img.shape[:2] == (self.nx * self.upscale, self.ny * self.upscale):
            print('Not upscaling background image')
            img = self.bg_img
        else:
            x_upscale = (self.nx * self.upscale) / self.bg_img.shape[0]
            y_upscale = (self.ny * self.upscale) / self.bg_img.shape[1]
            bg_rescale = np.mean([x_upscale, y_upscale])
            print(f'Resizing background image by factor {bg_rescale}')

            if (bg_rescale >= 1) and (int(bg_rescale) == bg_rescale):
                img = image_utils.upscale_image(self.bg_img, bg_rescale)
            else:
                img = image_utils.rescale_image(self.bg_img, bg_rescale)

        img = np.flipud(np.swapaxes(img, 0, 1))

        with hold_canvas(self.canvas_bg):
            self.canvas_bg.clear()
            self.canvas_bg.put_image_data(img, 0, 0)

    def update_all(self):
        self.update_shift_widget()
        self.update_roi_options()
        self.update_info()

    def draw_all(self, update=True):
        self.draw_bg()
        self.draw_current_mask_img(update=update)
        self.draw_roi_masks_img(update=update)

    @staticmethod
    def split_name(stim_condition):
        stim = stim_condition.split('(')[0]
        condition = stim_condition.split('(')[1].split(')')[0]
        return stim, condition

    def insert_database(self, roi_mask_tab, field_key):
        from djimaging.utils.mask_utils import to_igor_format, compare_roi_masks

        stim_to_roi_mask = {}
        for stim in self.stim_names:
            self.set_selected_stim(stim)
            roi_mask = self.prep_roi_mask_for_file()
            stim_to_roi_mask[stim] = to_igor_format(roi_mask)

        main_stim_condition = self.stim_names[self.main_stim_idx]
        main_roi_mask = stim_to_roi_mask[main_stim_condition]

        stim, condition = self.split_name(main_stim_condition)
        new_key = {**field_key, "stim_name": stim, "roi_mask": main_roi_mask, "condition": condition}

        roi_mask_tab().insert1(new_key)
        for i, (stim_condition, roi_mask) in enumerate(stim_to_roi_mask.items()):
            stim, condition = self.split_name(stim_condition)
            new_key = {**field_key, "stim_name": stim, "roi_mask": roi_mask, "condition": condition}

            as_field_mask, (shift_dx, shift_dy) = compare_roi_masks(roi_mask, main_roi_mask)
            new_key['as_field_mask'] = as_field_mask
            new_key['shift_dx'] = shift_dx
            new_key['shift_dy'] = shift_dy

            roi_mask_tab().RoiMaskPresentation().insert1(new_key)

    def create_widget_roi_next(self):
        widget = Button(description='>>', disabled=False, button_style='success',
                        layout=Layout(width='50px'))
        widget.on_click(self.exec_roi_next)
        return widget

    def exec_roi_next(self, button=None):
        next_roi = self.widget_roi.options[(self.widget_roi.index + 1) % len(self.widget_roi.options)]
        self.set_selected_roi(next_roi)

    def create_widget_roi_prev(self):
        widget = Button(description='<<', disabled=False, button_style='success',
                        layout=Layout(width='50px'))
        widget.on_click(self.exec_roi_prev)
        return widget

    def exec_roi_prev(self, button=None):
        prev_roi = self.widget_roi.options[(self.widget_roi.index - 1) % len(self.widget_roi.options)]
        self.set_selected_roi(prev_roi)

    def create_widget_diagnostics(self):
        return Output()
