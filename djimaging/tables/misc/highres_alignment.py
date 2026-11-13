"""
Alignment of the HighRes scan of a field to the field's ROI mask: shift and match score.

The HighRes scan is usually the same field at higher pixel resolution, recorded at the end of the stimulus session; it
can be slightly shifted in x, y and z. This table resamples the HR (``HighRes.StackAverages``) to the field grid and
compares it with the channel averages of the presentation the ROI mask was drawn on (``RoiMask``; the stored mask is
aligned to this presentation). Computation: ``djimaging.utils.highres_alignment``; validation: docs/motion_qc.md.

Example usage:

from djimaging.tables import misc

@schema
class HighResAlignment(misc.HighResAlignmentTemplate):
    highres_table = HighRes
    roi_mask_table = RoiMask
    presentation_table = Presentation
    userinfo_table = UserInfo
"""
from __future__ import annotations

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_storage import load_array
from djimaging.utils.dj_utils import get_primary_key


def _none_if_nan(x) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else x


class HighResAlignmentTemplate(dj.Computed):
    """Shift of the HighRes scan relative to the field ROI mask and how well it matches (incl. a rough z offset)."""
    database = ""
    _max_shift = 8  # search window of the registration [field px]

    @property
    def definition(self) -> str:
        return """
        # Alignment of the HighRes scan to the field ROI mask
        -> self.highres_table
        -> self.roi_mask_table
        ---
        shift_x : float32  # displacement of the HR vs. the ROI mask [field px]: mask position p is at p + shift in the HR
        shift_y : float32
        shift_x_um : float32  # the same in um
        shift_y_um : float32
        peak : float32  # height of the registration peak (registration image units)
        match_ch0=null : float32  # noise-corrected similarity after alignment, functional channel (1 = same structure)
        match_ch1=null : float32  # the same, structural channel (e.g. SR101; activity-independent)
        dz_um_approx=null : float32  # rough |z offset| from the match [um]; <~2 um: same plane within the precision
        mask_fit_hr=null : float32  # mask_utils.mask_fit_score of the ROI mask on the aligned HR (functional channel)
        mask_fit_ref=null : float32  # the same on the mask's own presentation (for comparison)
        scale : float32  # field pixel size / HR pixel size
        stage_dx_um=null : float32  # HR - mask presentation stage position [um]; not a reliable predictor of the shift
        stage_dy_um=null : float32
        stage_dz_um=null : float32
        hr_ch0 : <blob>  # HR functional channel resampled to the field grid, in the frame of the ROI mask
        hr_ch1=null : <blob>  # HR structural channel, the same
        """

    @property
    @abstractmethod
    def highres_table(self):
        pass

    @property
    @abstractmethod
    def roi_mask_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.highres_table.proj() * self.roi_mask_table.proj()
        except (AttributeError, TypeError):
            pass

    def _channel_names(self, key: dict) -> tuple[str, str | None]:
        """Functional channel (first functional) and structural channel (first other channel, e.g. SR101)."""
        functional = self.userinfo_table().get_functional_ch_names(key)
        others = [ch for ch in self.userinfo_table().get_ch_names(key) if ch not in functional]
        return functional[0], (others[0] if others else None)

    def _mask_presentation_key(self, key: dict) -> dict:
        row = (self.roi_mask_table & key).fetch1()
        return {k: row[k] for k in self.presentation_table.primary_key}

    def make(self, key: dict) -> None:
        from djimaging.utils.highres_alignment import align_highres
        func, struct = self._channel_names(key)
        roi_mask = load_array((self.roi_mask_table & key).fetch1('roi_mask'))
        pres_key = self._mask_presentation_key(key)
        pres = (self.presentation_table & pres_key).fetch1('pixel_size_um', 'npixartifact', 'absx', 'absy', 'absz')
        hr = (self.highres_table & key).fetch1('pixel_size_um', 'npixartifact', 'absx', 'absy', 'absz', 'scan_type')
        if hr[-1] != 'xy':
            raise ValueError(f"HighRes scan_type={hr[-1]!r} for {key}: only xy scans can be aligned.")

        def hr_avg(ch):
            rows = (self.highres_table.StackAverages & key & dict(ch_name=ch)).to_arrays('ch_average')
            return load_array(rows[0]) if len(rows) else None

        def pres_avg(ch):
            rows = (self.presentation_table.StackAverages & pres_key & dict(ch_name=ch)).to_arrays('ch_average')
            return load_array(rows[0]) if len(rows) else None

        hr0, ref0 = hr_avg(func), pres_avg(func)
        if hr0 is None or ref0 is None:
            raise ValueError(f"Missing channel average {func!r} of the HighRes or of the mask presentation for {key}.")
        hr1, ref1 = (hr_avg(struct), pres_avg(struct)) if struct is not None else (None, None)
        if hr1 is None or ref1 is None:
            hr1, ref1 = None, None

        a = align_highres(hr0, hr1, float(hr[0]), int(hr[1]), ref0, ref1, float(pres[0]), int(pres[1]),
                          roi_mask=roi_mask, max_shift=self._max_shift)
        self.insert1(dict(
            key, shift_x=float(a['shift'][0]), shift_y=float(a['shift'][1]), shift_x_um=float(a['shift_um'][0]),
            shift_y_um=float(a['shift_um'][1]), peak=float(a['peak']), match_ch0=_none_if_nan(a['match_ch0']),
            match_ch1=_none_if_nan(a['match_ch1']), dz_um_approx=_none_if_nan(a['dz_um_approx']),
            mask_fit_hr=_none_if_nan(a.get('mask_fit_hr', np.nan)),
            mask_fit_ref=_none_if_nan(a.get('mask_fit_ref', np.nan)), scale=float(a['scale']),
            stage_dx_um=_none_if_nan(hr[2] - pres[2]), stage_dy_um=_none_if_nan(hr[3] - pres[3]),
            stage_dz_um=_none_if_nan(hr[4] - pres[4]), hr_ch0=a['hr_ch0'].astype(np.float32),
            hr_ch1=None if a['hr_ch1'] is None else a['hr_ch1'].astype(np.float32)))

    def plot1(self, key: dict | None = None, gamma: float = 0.7) -> None:
        """Mask presentation, aligned HR (both channels) with the ROI mask outlines, and an overlay
        (mask presentation green, aligned HR magenta)."""
        from matplotlib import pyplot as plt
        key = get_primary_key(self, key)
        row = (self & key).fetch1()
        func, struct = self._channel_names(key)
        pres_key = self._mask_presentation_key(key)
        roi_mask = load_array((self.roi_mask_table & key).fetch1('roi_mask'))
        ref0 = load_array((self.presentation_table.StackAverages & pres_key & dict(ch_name=func)).fetch1('ch_average'))

        def norm(img):
            img = np.asarray(img, float)
            lo, hi = np.nanpercentile(img, [1, 99.5])
            return np.nan_to_num(np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1)) ** gamma

        imgs = [(f'mask presentation ({func})', norm(ref0)), (f'HR aligned ({func})', norm(load_array(row['hr_ch0'])))]
        if row['hr_ch1'] is not None:
            imgs.append((f'HR aligned ({struct})', norm(load_array(row['hr_ch1']))))
        fig, axs = plt.subplots(1, len(imgs) + 1, figsize=(4 * (len(imgs) + 1), 4.3))
        for ax, (title, img) in zip(axs, imgs):
            ax.imshow(img.T, cmap='gray', origin='lower')
            ax.contour((roi_mask < 0).T, levels=[0.5], colors='y', linewidths=0.6, origin='lower')
            ax.set_title(title, fontsize=9)
        overlay = np.stack([imgs[1][1], imgs[0][1], imgs[1][1]], axis=-1)
        axs[-1].imshow(np.transpose(overlay, (1, 0, 2)), origin='lower')
        axs[-1].set_title('green: mask presentation, magenta: HR', fontsize=9)
        m0 = 'n/a' if row['match_ch0'] is None else f"{row['match_ch0']:.2f}"
        m1 = 'n/a' if row['match_ch1'] is None else f"{row['match_ch1']:.2f}"
        dz = 'n/a' if row['dz_um_approx'] is None else f"~{row['dz_um_approx']:.1f} um"
        fig.suptitle(f"{key}\nshift ({row['shift_x']:.2f}, {row['shift_y']:.2f}) px, match {func} {m0}, {struct} {m1}, "
                     f"|dz| {dz}", fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot(self, restriction: dict | None = None) -> None:
        """Distributions of shift, match and approximate z offset."""
        from matplotlib import pyplot as plt
        restriction = dict() if restriction is None else restriction
        sx, sy, m0, m1, dz = (self & restriction).to_arrays('shift_x', 'shift_y', 'match_ch0', 'match_ch1',
                                                             'dz_um_approx')
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
        axs[0].scatter(sx, sy, s=8)
        axs[0].set(xlabel='shift x [px]', ylabel='shift y [px]', title='HR vs. ROI mask', aspect='equal')
        axs[1].hist([np.asarray(m0, float), np.asarray(m1, float)], bins=20, range=(0, 1), label=['ch0', 'ch1'])
        axs[1].set(xlabel='match', title='similarity after alignment')
        axs[1].legend()
        axs[2].hist(np.asarray(dz, float)[np.isfinite(np.asarray(dz, float))], bins=12)
        axs[2].set(xlabel='|dz| approx. [um]', title='rough z offset')
        plt.tight_layout()
        plt.show()
