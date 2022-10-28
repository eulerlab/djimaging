import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse
from skimage.transform import rotate, resize

from djimaging.tables.morphology import morphology_roi_utils
from djimaging.utils import math_utils


# TODO: Clean up this code? It is a mess and a miracle it does no break.

def match_rec_to_stack(
        ch_average, roi_mask, setup_xscale, wparams_rec, wparams_stack, pixel_sizes_stack, linestack,
        rec_cxy_stack, soma_xyz, soma_radius,
        angle_adjust=0, pad_scale=1.2, pad_more=50, shift=(0, 0), dist_score_factor=1e-3, rescales=None,
        blur_decay=5., blur_power=2., blur_factor=0.9, blur_postpower=0.5,
        template_threshold=False, template_max=0.8,
        savefilename=None, seed=42):

    """Get position of ROIs"""
    np.random.seed(seed)

    linestack = linestack.copy()
    if (soma_radius is not None) and (soma_xyz is not None):
        linestack = add_soma_to_linestack(
            linestack, pixel_sizes_stack, soma_xyz, radius_xyz=soma_radius)

    rec_rot = rotate_rec(
        ch_average=ch_average, wparams_rec=wparams_rec, wparams_stack=wparams_stack,
        setup_xscale=setup_xscale, angle_adjust=angle_adjust)

    roi_ids, _, roi_coords_rot = rotate_roi(
        roi_mask=roi_mask, wparams_rec=wparams_rec, wparams_stack=wparams_stack,
        setup_xscale=setup_xscale, angle_adjust=angle_adjust)

    crop_xy0, crop_xyz = get_crop(
        linestack=linestack, rec_shape=rec_rot.shape, rec_cxy=rec_cxy_stack,
        pixel_sizes_stack=pixel_sizes_stack, pad_scale=pad_scale, pad_more=pad_more, shift=shift)

    fit_dict = find_best_match(
        crop_xyz, rec_rot, roi_coords_rot=roi_coords_rot, rescales=rescales,
        dist_score_factor=dist_score_factor,
        exp_center=(rec_cxy_stack[0] - crop_xy0[0], rec_cxy_stack[1] - crop_xy0[1]),
        decay=blur_decay, power=blur_power, blur=blur_factor, postblurpower=blur_postpower,
        threshold=template_threshold, max_factor=template_max,
        plot_zidxs=False, savefilename=savefilename)

    rec_rot_cxy0 = np.array([rec_rot.shape[0] * fit_dict['rescale'] / 2,
                             rec_rot.shape[1] * fit_dict['rescale'] / 2])

    rec_cpos_stack_xyz = np.append(fit_dict['xyz_fit'][:2] + rec_rot_cxy0 + crop_xy0, fit_dict['xyz_fit'][2])
    rois_pos_stack_xy = roi_coords_rot * fit_dict['rescale'] + fit_dict['xyz_fit'][:2] + crop_xy0
    rois_pos_stack_xyz = np.pad(rois_pos_stack_xy, pad_width=((0, 0), (0, 1)), constant_values=fit_dict['xyz_fit'][2])

    return roi_ids, rois_pos_stack_xyz, rec_cpos_stack_xyz, fit_dict


def rotate_rec(ch_average, wparams_rec, wparams_stack, setup_xscale, angle_adjust=0):
    ang_deg = wparams_rec['Angle_deg'] + angle_adjust  # rotate angle (degree)

    assert wparams_stack['Angle_deg'] == 0.0, 'Not implemented for non zero rotation of stack'

    rec_resized = resize_rec(ch_average, wparams_rec, wparams_stack, setup_xscale)
    rec_rot = rotate(rec_resized, ang_deg, resize=True, order=1, cval=np.nan)

    return rec_rot


def get_roi_ids(rois):
    roi_ids = -np.unique(rois.astype(int))[:-1][::-1]
    return roi_ids


def rotate_roi(roi_mask, wparams_rec, wparams_stack, setup_xscale, angle_adjust=0, no_rois_lost=True):
    ang_deg = wparams_rec['Angle_deg'] + angle_adjust  # rotate angle (degree)

    rec_rois = resize_roi(roi_mask, wparams_rec, wparams_stack, setup_xscale)
    rec_rois_rot = rotate(rec_rois, ang_deg, cval=1, order=0, resize=True)
    if no_rois_lost:
        rec_rois_rot = _add_back_lost_rois(rec_rois=rec_rois, mod_rois=rec_rois_rot)

    rec_rois_rot = np.ma.masked_where(rec_rois_rot == 1, rec_rois_rot)

    (shift_x, shift_y) = 0.5 * np.array(rec_rois_rot.shape) - 0.5 * np.array(rec_rois.shape)
    (cx, cy) = 0.5 * np.array(rec_rois.shape)

    roi_ids = get_roi_ids(roi_mask)

    # reverse the labels to keep consistent with the labels of raw traces
    px = [np.vstack(np.where(-rec_rois == roi_id)).T[:, 0].mean() for roi_id in roi_ids]
    py = [np.vstack(np.where(-rec_rois == roi_id)).T[:, 1].mean() for roi_id in roi_ids]

    xn, yn = rotate_point_around_center(px, py, cx, cy, shift_x, shift_y, ang_deg)

    return roi_ids, rec_rois_rot, np.vstack([xn, yn]).T


def rotate_point_around_center(px, py, cx, cy, shift_x, shift_y, ang_deg):
    ang_rad = ang_deg * np.pi / 180.  # rotate angle (radian)

    px -= cx
    py -= cy

    xn = px * np.cos(ang_rad) - py * np.sin(ang_rad)
    yn = px * np.sin(ang_rad) + py * np.cos(ang_rad)

    xn += (cx + shift_x)
    yn += (cy + shift_y)

    return xn, yn


def get_crop(linestack, rec_shape, rec_cxy, pixel_sizes_stack, pad_scale=1., pad_more=50, shift=(0, 0)):
    sizex = rec_shape[0] * pad_scale + pad_more / pixel_sizes_stack[0]
    sizey = rec_shape[1] * pad_scale + pad_more / pixel_sizes_stack[1]

    crop_x0, crop_y0, crop_x1, crop_y1 = get_crop_idxs(linestack, rec_cxy[0], rec_cxy[1], sizex, sizey, shift)
    crop_xy0 = np.array([crop_x0, crop_y0])
    crop_xyz = linestack[crop_x0:crop_x1, crop_y0:crop_y1, :]

    return crop_xy0, crop_xyz


def get_rec_center_in_stack_coordinates(wparams_rec, wparams_stack, linestack, pixel_sizes_stack):
    """Get center position of rec"""

    # TODO: Why not use d_rel_cz?

    d_rel_cy, d_rel_cx, d_rel_cz = rel_position_um(wparams_stack, wparams_rec) / pixel_sizes_stack
    cx = linestack.shape[0] / 2.
    cy = linestack.shape[1] / 2.

    warning_flag = 0

    rec_cx_stack_coords = int(cx + d_rel_cx)
    rec_cy_stack_coords = int(cy + d_rel_cy)

    if (rec_cx_stack_coords < 0) or (rec_cx_stack_coords >= linestack.shape[0]):
        warnings.warn(f"rec_cx_stack_coords={rec_cx_stack_coords} is outside stack (0, {linestack.shape[0]}).")
        rec_cx_stack_coords = int(np.clip(rec_cx_stack_coords, 0, linestack.shape[0] - 1))
        warning_flag = 1

    if (rec_cy_stack_coords < 0) or (rec_cy_stack_coords >= linestack.shape[1]):
        warnings.warn(f"rec_cy_stack_coords={rec_cy_stack_coords} is outside stack (0, {linestack.shape[1]}).")
        rec_cy_stack_coords = int(np.clip(rec_cy_stack_coords, 0, linestack.shape[1] - 1))
        warning_flag = 1

    return rec_cx_stack_coords, rec_cy_stack_coords, warning_flag


def rel_position_um(d1_wparams_num, d2_wparams_num):
    """Relative position between two datasets. Typically, soma and dendrites"""
    pos1 = position_um(d1_wparams_num)
    pos2 = position_um(d2_wparams_num)
    return pos1 - pos2


def position_um(wparams_num):
    """Get position of recording"""
    x = wparams_num["XCoord_um"]
    y = wparams_num["YCoord_um"]
    z = wparams_num["ZCoord_um"]
    return np.array([x, y, z])


def resize_rec(ch_average, wparams_rec, wparams_num, setup_xscale):
    output_shape = np.ceil(
        np.asarray(ch_average.shape) * get_scale_factor_xy(wparams_rec, wparams_num, setup_xscale)).astype(int)
    return resize(ch_average, output_shape=output_shape, order=0, mode='constant')


def create_template(rec, npixartifact, data_stack_name):

    data = None
    if 'ROI_CorrMatrix' in rec:
        data = rec['ROI_CorrMatrix']
        if np.std(data[npixartifact:, :]) == 0:
            data = None

    if data is None:
        data = rec[data_stack_name]

    data_template = np.mean(data, axis=2)
    data_template[npixartifact:, :] = math_utils.normalize_zscore(data_template[npixartifact:, :])
    data_template[:npixartifact, :] = np.nan

    return data_template


def resize_image(image, scale, dtype="int", order=0):
    output_shape = np.ceil(np.asarray(image.shape) * scale).astype(dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized_image = resize(image, output_shape=output_shape, order=order, mode='constant', anti_aliasing=False)
    return resized_image


def resize_roi(roi_mask, wparams_rec, wparams_stack, setup_xscale, no_rois_lost=True):
    """Rescale ROIs"""
    output = np.round(resize_image(
        image=roi_mask, scale=get_scale_factor_xy(wparams_rec, wparams_stack, setup_xscale), dtype="int"))
    if no_rois_lost:
        output = _add_back_lost_rois(rec_rois=roi_mask, mod_rois=output)

    return output


def _add_back_lost_rois(rec_rois, mod_rois):
    """If ROIs where lost during rescaling, add them back as a single pixel"""

    mod_rois = mod_rois.copy()

    input_rois = set(rec_rois[rec_rois < 0])
    output_rois = set(mod_rois[mod_rois < 0])

    lost_rois = input_rois.difference(output_rois)  # ROIs that are lost in rescaling

    for lost_roi in lost_rois:
        xs, ys = np.where(rec_rois == lost_roi)
        _xs = xs.copy()
        _ys = ys.copy()

        for i in range(10):
            if i > 0:
                warnings.warn(f'Expanding search space to i={i}')
                xoff = np.concatenate(
                    [np.arange(-i, i + 1), np.full(2 * i - 1, i), np.arange(-i, i + 1), np.full(2 * i - 1, -i)])
                yoff = np.concatenate(
                    [np.full(2 * i + 1, i), np.arange(-i + 1, i), np.full(2 * i + 1, -i), np.arange(-i + 1, i)])

                for xi, yi in zip(xs, ys):
                    _xs = np.append(_xs, xi + xoff)
                    _ys = np.append(_ys, yi + yoff)

            xy_out = _find_best_pixel_to_add_roi_back(rec_rois, mod_rois, xs=_xs, ys=_ys)

            if xy_out is not None:
                mod_rois[xy_out[0], xy_out[1]] = lost_roi
                break

    assert np.all(np.unique(mod_rois) == np.unique(rec_rois)), f'{np.unique(mod_rois)} vs. {np.unique(rec_rois)}'

    return mod_rois


def get_scale_factor_xy(wparams_rec, wparams_stack, setup_xscale):
    """
    get the scale factor from rec to wparams_num,
    e.g. scipy.misc.im
    (rec, size=scale_factor, interp='nearest')
    would make the rec into the same scale as wparams_num.
    """

    rec_pixel_size = get_pixel_size_um_rec(wparams_rec, setup_xscale)[0]
    stack_pixel_sizes = get_pixel_size_um_rec(wparams_stack, setup_xscale)[0]

    return rec_pixel_size / stack_pixel_sizes


def _find_best_pixel_to_add_roi_back(rois, output, xs, ys):
    # Start with pixel closest to center
    mdists = (xs - np.mean(xs)) ** 2 + (ys - np.mean(ys)) ** 2
    dist_idx = np.argsort(mdists)

    for xi, yi in zip(xs[dist_idx], ys[dist_idx]):
        xi_out = np.min([int(np.round(xi * output.shape[0] / rois.shape[0])), output.shape[0] - 1])
        yi_out = np.min([int(np.round(yi * output.shape[1] / rois.shape[1])), output.shape[1] - 1])

        # Accept pixel only if it does not overwrite another ROI
        if np.sum(output == output[xi_out, yi_out]) > 1:
            return xi_out, yi_out

    return None


def get_pixel_size_um_rec(wparams_num, setup_xscale):
    """Return the real length (in um) of each pixel point."""

    zoom = wparams_num.get('Zoom', wparams_num.get('zoom', np.nan))

    nxpix_all = wparams_num.get('User_dxPix', wparams_num.get('user_dxpix', np.nan))
    nxpix_retrace = wparams_num.get('User_nPixRetrace', wparams_num.get('user_npixretrace', np.nan))
    nxpix_lineoffset = wparams_num.get('User_nXPixLineOffs', wparams_num.get('user_nxpixlineoffs', np.nan))

    nxpix = nxpix_all - nxpix_retrace - nxpix_lineoffset

    assert nxpix > 0, wparams_num.keys()

    len_stack_x_um = setup_xscale / zoom
    stack_pixel_size_x = len_stack_x_um / nxpix
    stack_pixel_size_y = stack_pixel_size_x

    stack_pixel_size_z = wparams_num.get('ZStep_um', wparams_num.get('zstep_um', np.nan))

    return np.array([stack_pixel_size_x, stack_pixel_size_y, stack_pixel_size_z])


def blur_linestack(linestack, blur, postblurpower=None, inplace=False):
    assert np.min(linestack) == 0
    assert np.max(linestack) == 1
    if not inplace:
        linestack = linestack.copy()

    max_unblurred = np.max(linestack)
    linestack = gaussian_filter(linestack.astype('float32'), sigma=blur)
    linestack /= np.max(linestack)  # Renormalize
    if postblurpower is not None:
        linestack = linestack ** postblurpower
    linestack *= max_unblurred
    return linestack


def get_weighted_linestack_xy(zidx, linestack, decay=5., power=2., plot=False):
    assert linestack.ndim == 3
    assert 0 <= zidx <= linestack.shape[2] - 1

    linestack = linestack
    zdists = np.concatenate([np.arange(zidx, 0, -1), np.arange(0, linestack.shape[2] - zidx)])
    zweights = np.exp(-np.abs(zdists / decay) ** power)
    weighted_linestack_xy = np.max(linestack * zweights, axis=2)

    if plot:
        plot_weighted_linestack_xy(
            zdists=zdists, zweights=zweights, linestack_slice_xy=linestack[:, :, zidx],
            weighted_linestack_xy=weighted_linestack_xy)

    return weighted_linestack_xy


def plot_weighted_linestack_xy(zdists, zweights, linestack_slice_xy, weighted_linestack_xy):
    fig, axs = plt.subplots(1, 4, figsize=(12, 5))
    axs[0].plot(zdists)
    axs[1].plot(zweights)
    im = axs[2].imshow(linestack_slice_xy, cmap='Greys')
    plt.colorbar(im, ax=axs[2])
    im = axs[3].imshow(weighted_linestack_xy)
    plt.colorbar(im, ax=axs[3])
    plt.show()


def match_template_to_image(image, template, exp_center, dist_score_factor=1e-3):
    """Find template in image"""
    np.random.seed(42)

    htemplate, wtemplate = template.shape

    image = image.copy().astype("float32")
    template = template.copy().astype("float32")
    mask = np.isfinite(template).astype("float32")
    match_score = np.zeros(shape=[image.shape[i] - template.shape[i] + 1 for i in range(2)], dtype=image.dtype)
    match_score = cv2.matchTemplate(
        image, np.nan_to_num(template, copy=True), method=cv2.TM_SQDIFF, result=match_score, mask=mask)
    match_score *= -100. / (template.shape[0] * template.shape[1])

    xv, yv = np.meshgrid(
        np.arange(match_score.shape[1]),
        np.arange(match_score.shape[0]),
    )

    dist_score = -((xv - exp_center[0] + wtemplate // 2) ** 2 + (yv - exp_center[1] + htemplate // 2) ** 2)

    score_map = dist_score_factor * dist_score + match_score

    best_score = np.max(score_map)
    y, x = np.unravel_index(np.argmax(score_map), score_map.shape)
    best_pos = x, y

    return best_score, best_pos, score_map


def plot_match_template_to_image(
        image, template_fit, score_map, best_xy, best_score,
        template_raw=None, roi_coords_rot=None, exp_center=None,
        savefilename=None):
    htemplate, wtemplate = template_fit.shape
    x, y = best_xy

    zoomextraw = 0.3 * np.max([htemplate, wtemplate])

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title('original')

    if template_raw is not None:
        im = ax.imshow(template_raw, cmap='rainbow', origin='lower')
        plt.colorbar(im, ax=ax, label='level')
        ax.set_xlim(axs[0].get_xlim())

    if roi_coords_rot is not None:
        for i, (roix, roiy) in enumerate(roi_coords_rot, start=1):
            ax.scatter(roix, roiy, marker='o', color='none', ec='k', s=150, alpha=0.8)
            ax.scatter(roix, roiy, marker='o', color='white', ec='none', s=150, alpha=0.4)
            ax.text(roix, roiy, i, color='k', ha='center', va='center')

    ax = axs[1]
    im = ax.imshow(template_fit, cmap='rainbow', origin='lower', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='level')
    ax.set_title('template')
    ax.set_xlim(axs[0].get_xlim())

    ax = axs[2]
    im = ax.imshow(image, cmap='rainbow', origin='lower', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='level')
    ax.set_title('crop')
    ax.plot([x, x + wtemplate, x + wtemplate, x, x], [y, y, y + htemplate, y + htemplate, y], c='r')
    ax.set(xlim=(0, image.shape[1]), ylim=(0, image.shape[0]))

    if exp_center is not None:
        ax.plot([exp_center[0], x + wtemplate / 2], [exp_center[1], y + htemplate / 2],
                '-x', c='k', zorder=100)

    ax = axs[3]
    im = ax.imshow(score_map, cmap='rainbow', origin='lower', vmin=np.min(score_map), vmax=np.max(score_map),
                   extent=(
                       wtemplate // 2, score_map.shape[1] + wtemplate // 2,
                       htemplate // 2, score_map.shape[0] + htemplate // 2,
                   ))
    plt.colorbar(im, ax=ax, label='score')
    ax.set_title("match scores")
    ax.plot([x, x + wtemplate, x + wtemplate, x, x], [y, y, y + htemplate, y + htemplate, y], c='k')
    ax.set(xlim=(0, image.shape[1]), ylim=(0, image.shape[0]))

    ax.plot(x + wtemplate // 2, y + htemplate // 2, 'o', markeredgecolor='purple', markerfacecolor='none',
            markersize=10)

    if exp_center is not None:
        ax.plot(exp_center[0], exp_center[1], '*', c='k', zorder=100)

    for ax in axs[4:6]:
        im = ax.imshow(image, cmap='rainbow', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=ax, label='level')
        ax.set_title(f'zoom - score={best_score:.3g}')
        ax.plot([x, x + wtemplate, x + wtemplate, x, x], [y, y, y + htemplate, y + htemplate, y], c='r')
        ax.set(xlim=(np.maximum(x - zoomextraw, 0), x + wtemplate + zoomextraw),
               ylim=(np.maximum(y - zoomextraw, 0), y + htemplate + zoomextraw))

    ax = axs[-1]
    template_fit = template_fit.copy().astype('float')
    if np.any(~np.isfinite(template_fit)):
        ax.contour(np.isfinite(template_fit), levels=[0.5], zorder=10, colors=['pink'], linestyles=[':'],
                   extent=(x, x + wtemplate, y, y + htemplate))
    with np.errstate(invalid='ignore'):
        template_fit[np.isfinite(template_fit) & np.less(template_fit, 0.25*np.nanmean(template_fit))] = np.nan
    ax.imshow(template_fit, cmap='Greys', extent=(x, x + wtemplate, y, y + htemplate), alpha=0.75, vmin=-1, vmax=1,
              origin='lower', zorder=1000)
    if roi_coords_rot is not None:
        ax.scatter(x + roi_coords_rot[:, 0], y + roi_coords_rot[:, 1], marker='o', color='none', ec='k',
                   s=100, alpha=1)

    fig.suptitle('recording matching to morphology', y=1, va='bottom')
    plt.tight_layout()

    if savefilename is not None:
        fig.savefig(savefilename, facecolor='white')

    plt.show()


def find_best_match(
        linestack, rec_rot, exp_center=None, decay=5., power=2., blur=1., postblurpower=0.5,
        rescales=None, threshold=True, dist_score_factor=1e-3, max_factor=0.75,
        roi_coords_rot=None, plot_zidxs=False, plot_steps=False, plot_result=True, savefilename=None):

    linestack_sums = np.sum(linestack, axis=(0, 1))
    zidxs = np.where(linestack_sums > np.median(linestack_sums))[0]

    if plot_zidxs:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(linestack_sums)
        ax.plot(zidxs, np.zeros_like(zidxs), '*', label='considered')
        ax.set(ylabel='density', xlabel='zidx')
        ax.legend()
        plt.show()

    best_score = None
    best_score_map = None
    best_linestack_xy = None
    best_xyz = None
    best_rescale = None
    best_template_fit = None
    best_template_raw = None

    if blur is not None:
        linestack = blur_linestack(linestack, blur=blur, postblurpower=postblurpower, inplace=False)

    if rescales is None:
        rescales = [1.0]

    for rescale in np.unique(rescales):
        template_raw = resize_image(rec_rot.copy(), rescale, dtype="float32", order=0).T

        if threshold:
            template_fit = np.zeros(template_raw.shape, dtype='float32')
            template_fit[(np.nan_to_num(template_raw) >
                         np.nanmedian(template_raw) + 0.25 * np.nanstd(template_raw))
                        & np.isfinite(template_raw)] = 1.0
            template_fit[~np.isfinite(template_raw)] = np.nan
        else:
            lsmin = np.nanpercentile(template_raw, 20)
            template_fit = template_raw - lsmin

            lsmax = np.nanpercentile(template_fit, 90)
            lmax = np.nanmax(template_fit)

            if lsmax > 0.1:
                template_fit /= lsmax
            else:
                template_fit /= lmax

            template_fit = np.clip(template_fit, 0., 1.)

        template_fit *= max_factor

        for zidx in zidxs:

            weighted_linestack_xy = get_weighted_linestack_xy(
                zidx=zidx, linestack=linestack, decay=decay, power=power, plot=False).T

            score, pos, score_map = match_template_to_image(
                image=weighted_linestack_xy, template=template_fit,
                exp_center=exp_center, dist_score_factor=dist_score_factor)

            if plot_steps:
                plot_match_template_to_image(
                    image=weighted_linestack_xy, template_fit=template_fit,
                    score_map=score_map, best_xy=pos, best_score=score,
                    template_raw=template_raw, roi_coords_rot=roi_coords_rot,
                    exp_center=exp_center, savefilename=None)

            if best_score is None or score > best_score:
                best_score = score
                best_score_map = score_map
                best_linestack_xy = weighted_linestack_xy
                best_xyz = pos[0], pos[1], zidx
                best_rescale = rescale
                best_template_fit = template_fit
                best_template_raw = template_raw

    if plot_result:
        plot_match_template_to_image(
            image=best_linestack_xy, template_fit=best_template_fit,
            score_map=best_score_map, best_xy=best_xyz[:2], best_score=best_score,
            template_raw=best_template_raw, roi_coords_rot=roi_coords_rot,
            exp_center=exp_center, savefilename=savefilename)

    best_fit_dict = dict(
        score=best_score,
        score_map=np.asarray(best_score_map).astype(np.float32),
        linestack_xy=np.asarray(best_linestack_xy).astype(np.float32),
        xyz_fit=np.asarray(best_xyz).astype(np.float32),
        exp_center=np.asarray(exp_center).astype(np.float32),
        template_fit=np.asarray(best_template_fit).astype(np.float32),
        template_raw=np.asarray(best_template_raw).astype(np.float32),
        rescale=best_rescale,
        roi_coords_rot=roi_coords_rot,
    )

    return best_fit_dict


def get_crop_idxs(linestack, rec_cx, rec_cy, sizex, sizey, shift=(0, 0)):
    """Get boundary indexes for crop"""
    # Get crop indexes relative to whole stack
    crop_x0 = np.maximum(rec_cx - sizex / 2 + shift[0], 0)
    crop_y0 = np.maximum(rec_cy - sizey / 2 + shift[1], 0)

    crop_x1 = np.minimum(rec_cx + sizex / 2 + shift[0], linestack.shape[0])
    crop_y1 = np.minimum(rec_cy + sizey / 2 + shift[1], linestack.shape[1])

    crop_x0, crop_y0, crop_x1, crop_y1 = \
        int(np.floor(crop_x0)), int(np.floor(crop_y0)), int(np.ceil(crop_x1)), int(np.ceil(crop_y1))

    return crop_x0, crop_y0, crop_x1, crop_y1


def add_soma_to_linestack(linestack, pixel_sizes_stack, soma, radius_xyz, fill_value=0.05):
    linestack = linestack.copy().astype(float)

    if isinstance(radius_xyz, (int, float)):
        radius_xyz = np.array([radius_xyz, radius_xyz, radius_xyz]).astype(np.float32)

    # Define
    radius_x = radius_xyz[0]
    radius_y = radius_xyz[1]
    radius_z = radius_xyz[2]

    # Compute stack indices
    ix = int(np.round(soma[0] / pixel_sizes_stack[0]))
    iy = int(np.round(soma[1] / pixel_sizes_stack[1]))
    iz = int(np.round(soma[2] / pixel_sizes_stack[2]))

    pixel_radius_x = int(np.round(radius_x / pixel_sizes_stack[0]))
    pixel_radius_y = int(np.round(radius_y / pixel_sizes_stack[1]))
    pixel_radius_z = int(np.round(radius_z / pixel_sizes_stack[2]))

    for jz in range(-pixel_radius_z + 1, pixel_radius_z):
        dxs, dys = ellipse(
            ix, iy,
            np.sqrt(pixel_radius_x ** 2 - (float(jz / pixel_radius_z) * pixel_radius_x) ** 2),
            np.sqrt(pixel_radius_y ** 2 - (float(jz / pixel_radius_z) * pixel_radius_x) ** 2),
            shape=(linestack.shape[0], linestack.shape[1]), rotation=np.deg2rad(0))

        linestack[dxs, dys, iz + jz] = fill_value

    return linestack


def find_roi_pos_stack(roi_raw_xyz, linestack_coords_xyz, soma_linestack_coords_xyz, max_dist, z_factor):
    pixel_stack_xyz, dist = calibrate_one_roi(
        roi_raw_xyz, linestack_coords_xyz, z_factor=z_factor, return_dist=True)

    _, dist_soma = calibrate_one_roi(
        roi_raw_xyz, soma_linestack_coords_xyz, z_factor=z_factor, return_dist=True)

    dist_to_morph = np.minimum(dist, dist_soma)

    if dist_to_morph <= max_dist:
        quality = True
        if dist_soma > dist:
            roi_cal_pos_stack_xyz = pixel_stack_xyz
            soma_by_dist = False
        else:
            roi_cal_pos_stack_xyz = np.nan
            soma_by_dist = True
    else:
        quality = False
        soma_by_dist = False
        roi_cal_pos_stack_xyz = pixel_stack_xyz

    return quality, roi_cal_pos_stack_xyz, soma_by_dist


def calibrate_one_roi(coords_xyz, linestack_coords_xyz, z_factor=1., return_dist=False):
    dists_xyz = ((coords_xyz - linestack_coords_xyz) * np.array([1., 1., z_factor]))
    dists_ec = np.sqrt(np.sum(dists_xyz ** 2, axis=1))
    best_dist_idx = np.argmin(dists_ec)
    best_linestack_coords_xyz = linestack_coords_xyz[best_dist_idx]
    if return_dist:
        return best_linestack_coords_xyz, dists_ec[best_dist_idx]
    else:
        return best_linestack_coords_xyz


def compute_roi_pos_metrics(roi_pos_stack_xyz, df_paths, soma_xyz, density_center, max_d_dist):
    roi_metrics = dict()

    # Find the path each ROI on
    path_id = morphology_roi_utils.on_which_path(df_paths, roi_pos_stack_xyz)
    roi_metrics['path_id'] = path_id

    # Find the location of each ROI on its corresponding path
    loc_on_path = morphology_roi_utils.get_loc_on_path_stack(df_paths, roi_pos_stack_xyz)
    roi_metrics['loc_on_path'] = loc_on_path

    # Find ROI pos in real length coordinate. Avoid converting by voxel size due to unavoidable rounding.
    roi_pos_xyz = df_paths.loc[path_id].path[loc_on_path]
    roi_metrics['roi_pos_xyz'] = roi_pos_xyz
    roi_metrics['roi_pos_xyz_rel_to_soma'] = roi_pos_xyz - soma_xyz

    # Get dendritic distance from ROI to soma
    d_dist_to_soma = morphology_roi_utils.compute_dendritic_distance_to_soma(df_paths, path_id, loc_on_path)
    roi_metrics['d_dist_to_soma'] = d_dist_to_soma
    roi_metrics['norm_d_dist_to_soma'] = d_dist_to_soma / max_d_dist

    # Get euclidean distance from ROI to soma
    ec_dist_to_soma = morphology_roi_utils.compute_euclidean_distance(roi_pos_xyz, soma_xyz)
    roi_metrics['ec_dist_to_soma'] = ec_dist_to_soma

    # Get euclidean distance from ROI to dendritic density center
    ec_dist_to_density_center = morphology_roi_utils.compute_euclidean_distance(roi_pos_xyz[:2], density_center)
    roi_metrics['ec_dist_to_density_center'] = ec_dist_to_density_center

    # Get number of branch points from ROI to soma
    branches_to_soma = np.array(df_paths.loc[path_id]['back_to_soma'], dtype='int')
    roi_metrics['branches_to_soma'] = branches_to_soma

    roi_metrics['num_branches_to_soma'] = len(branches_to_soma)

    return roi_metrics
