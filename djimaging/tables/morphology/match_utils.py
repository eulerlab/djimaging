import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse
from skimage.transform import rotate, resize


# TODO: Clean up this code? It is a mess and a miracle it does no break.

def match_rec_to_stack(
        rec, linestack, wparamsnum_stack, setup_xscale, pixel_sizes_stack,
        shift=(0, 0), pad_scale=1., pad_more=100, angle_adjust=0, dist_score_factor=1e-3,
        rescales=None, seed=42, savefilename=None):
    """Finetune placement of ROIs on morph. May require manual parameters."""

    np.random.seed(seed)

    roi_ids, rois_pos_stack_xyz, rec_cpos_stack_xyz = get_roi_coordinates(
        rec=rec, wparamsnum_stack=wparamsnum_stack, linestack=linestack,
        setup_xscale=setup_xscale, pixel_sizes_stack=pixel_sizes_stack,
        pad_scale=pad_scale, pad_more=pad_more, angle_adjust=angle_adjust,
        shift=shift, rescales=rescales, dist_score_factor=dist_score_factor, savefilename=savefilename)

    return roi_ids, rois_pos_stack_xyz, rec_cpos_stack_xyz


def get_roi_coordinates(rec, setup_xscale, wparamsnum_stack, pixel_sizes_stack, linestack,
                        angle_adjust=0, pad_scale=1.0, pad_more=50, shift=(0, 0),
                        dist_score_factor=1e-3, rescales=None, savefilename=None):
    """Get position of ROIs"""
    d_rec_rot = rotate_rec(rec, wparamsnum_stack, setup_xscale, angle_adjust)

    roi_ids, _, roi_coords_rot = rotate_roi(rec, wparamsnum_stack, setup_xscale, angle_adjust)

    rec_cx_stack, rec_cy_stack = get_rec_center_in_stack_coordinates(
        rec=rec, pixel_sizes_stack=pixel_sizes_stack, wparamsnum_stack=wparamsnum_stack, linestack=linestack)

    d_rec_rot_xyz0, d_rec_rescale, crop_xy0 = get_rec_pos(
        rec_rot=d_rec_rot, roi_coords_rot=roi_coords_rot,
        pixel_sizes_stack=pixel_sizes_stack,
        rec_cx_stack=rec_cx_stack, rec_cy_stack=rec_cy_stack,
        rescales=rescales, dist_score_factor=dist_score_factor, linestack=linestack,
        pad_scale=pad_scale, pad_more=pad_more, shift=shift, savefilename=savefilename)

    d_rec_rot_cxy0 = np.array([d_rec_rot.shape[0] * d_rec_rescale / 2, d_rec_rot.shape[1] * d_rec_rescale / 2])

    rec_cpos_stack_xy = d_rec_rot_cxy0 + d_rec_rot_xyz0[:2] + crop_xy0
    rec_cpos_stack_xyz = np.append(rec_cpos_stack_xy, d_rec_rot_xyz0[2])

    rois_pos_stack_xy = roi_coords_rot * d_rec_rescale + d_rec_rot_xyz0[:2] + crop_xy0
    rois_pos_stack_xyz = np.pad(rois_pos_stack_xy, pad_width=((0, 0), (0, 1)), constant_values=d_rec_rot_xyz0[2])

    return roi_ids, rois_pos_stack_xyz, rec_cpos_stack_xyz


def rotate_rec(rec, w_params_num_stack, setup_xscale, angle_adjust=0):
    ang_deg = rec['wParamsNum']['Angle_deg'] + angle_adjust  # rotate angle (degree)

    assert w_params_num_stack['Angle_deg'] == 0.0, 'Not implemented for non zero rotation of stack'

    rec_resized = resize_rec(rec, w_params_num_stack, setup_xscale)
    rec_rot = rotate(rec_resized, ang_deg, resize=True, order=1, cval=np.nan)

    return rec_rot


def get_roi_ids(rec):
    roi_ids = -np.unique(rec['ROIs'].astype(int))[:-1][::-1]
    return roi_ids


def rotate_roi(rec, w_params_stack, setup_xscale, angle_adjust=0, no_rois_lost=True):
    w_params_rec = rec['wParamsNum']

    ang_deg = w_params_rec['Angle_deg'] + angle_adjust  # rotate angle (degree)

    rec_rois = resize_roi(rec['ROIs'], w_params_rec, w_params_stack, setup_xscale)
    rec_rois_rot = rotate(rec_rois, ang_deg, cval=1, order=0, resize=True)
    if no_rois_lost:
        rec_rois_rot = _add_back_lost_rois(rec_rois=rec_rois, mod_rois=rec_rois_rot)

    rec_rois_rot = np.ma.masked_where(rec_rois_rot == 1, rec_rois_rot)

    (shift_x, shift_y) = 0.5 * np.array(rec_rois_rot.shape) - 0.5 * np.array(rec_rois.shape)
    (cx, cy) = 0.5 * np.array(rec_rois.shape)

    roi_ids = get_roi_ids(rec)

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


def get_rec_pos(rec_rot, pixel_sizes_stack, linestack, rec_cx_stack, rec_cy_stack,
                soma_xyz=None, soma_xyz_radius_xyz=None,
                roi_coords_rot=None, pad_scale=1.0, pad_more=50, shift=(0, 0),
                dist_score_factor=1e-3, rescales=None, savefilename=None):
    """Get recording position"""
    # Get crop
    sizex = rec_rot.shape[0] * pad_scale + pad_more / pixel_sizes_stack[0]
    sizey = rec_rot.shape[1] * pad_scale + pad_more / pixel_sizes_stack[1]

    crop_x0, crop_y0, crop_x1, crop_y1 = get_crop_idxs(
        linestack, rec_cx_stack, rec_cy_stack, sizex, sizey, shift)
    crop_xy0 = np.array([crop_x0, crop_y0])

    if (soma_xyz_radius_xyz is not None) and (soma_xyz is not None):
        linestack = add_soma_to_linestack(
            linestack, pixel_sizes_stack, soma_xyz, radius_xyz=soma_xyz_radius_xyz)

    crop_xyz = linestack[crop_x0:crop_x1, crop_y0:crop_y1, :]

    d_rec_rot_xyz0, d_rec_rescale = find_best_match(
        crop_xyz, rec_rot, roi_coords_rot=roi_coords_rot, threshold=False, rescales=rescales,
        dist_score_factor=dist_score_factor,
        exp_center=(rec_cx_stack - crop_x0, rec_cy_stack - crop_y0),
        plot_zidxs=False, savefilename=savefilename)

    return d_rec_rot_xyz0, d_rec_rescale, crop_xy0


def get_rec_center_in_stack_coordinates(rec, pixel_sizes_stack, wparamsnum_stack, linestack):
    """Get center position of rec"""

    d_rel_cy, d_rel_cx, _ = rel_position_um(wparamsnum_stack, rec['wParamsNum']) / pixel_sizes_stack[0]
    cx = linestack.shape[0] / 2.
    cy = linestack.shape[1] / 2.

    rec_cx_stack_coords, rec_cy_stack_coords = int(cx + d_rel_cx), int(cy + d_rel_cy)

    if (rec_cx_stack_coords < 0) or (rec_cx_stack_coords >= linestack.shape[0]):
        warnings.warn(
            f"rec_cx_stack_coords is broken: rec_cx_stack_coords={rec_cx_stack_coords}.\n" + \
            f"Set to stack border. \nd_rel_cx={d_rel_cx} d_rel_cy={d_rel_cy}")

        rec_cx_stack_coords = int(np.clip(rec_cx_stack_coords, 0, linestack.shape[0] - 1))

    if (rec_cy_stack_coords < 0) or (rec_cy_stack_coords >= linestack.shape[1]):
        warnings.warn(
            f"rec_cy_stack_coords is broken: rec_cy_stack_coords={rec_cy_stack_coords}. Set to stack border.\n" + \
            f"d_rel_cx={d_rel_cx:.1f} d_rel_cy={d_rel_cy:.1f}")
        rec_cy_stack_coords = int(np.clip(rec_cy_stack_coords, 0, linestack.shape[0] - 1))

    return rec_cx_stack_coords, rec_cy_stack_coords


def rel_position_um(d1_w_params_num, d2_w_params_num):
    """Relative position between two datasets. Typically, soma and dendrites"""
    pos1 = position_um(d1_w_params_num)
    pos2 = position_um(d2_w_params_num)
    return pos1 - pos2


def position_um(w_params_num):
    """Get position of recording"""
    x = w_params_num["XCoord_um"]
    y = w_params_num["YCoord_um"]
    z = w_params_num["ZCoord_um"]
    return np.array([x, y, z])


def resize_rec(rec, w_params_num, setup_xscale):
    reci = rec_preprop(rec)
    output_shape = np.ceil(
        np.asarray(reci.shape) * get_scale_factor_xy(rec['wParamsNum'], w_params_num, setup_xscale)).astype(int)
    return resize(reci, output_shape=output_shape, order=0, mode='constant')


def rec_preprop(rec):
    if 'ROI_CorrMatrix' in rec.keys():
        rec = rec['ROI_CorrMatrix'].mean(2)
        rec[:2, :] = rec.mean() - 0.5 * rec.std()
    else:
        rec = rec['wDataCh0'].mean(2)
        rec[4:, :] = (rec[4:, :] - np.mean(rec[4:, :])) / np.std(rec[4:, :])
        rec[:4, :] = np.min(rec[4:, :])

    return rec


def resize_image(image, scale, dtype="int", order=0):
    output_shape = np.ceil(np.asarray(image.shape) * scale).astype(dtype)
    return resize(image, output_shape=output_shape, order=order, mode='constant', anti_aliasing=False)


def resize_roi(roi_mask, w_params_rec, w_params_stack, setup_xscale, no_rois_lost=True):
    """Rescale ROIs"""
    output = np.round(resize_image(
        image=roi_mask, scale=get_scale_factor_xy(w_params_rec, w_params_stack, setup_xscale), dtype="int"))
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


def get_scale_factor_xy(w_params_rec, w_params_stack, setup_xscale):
    """
    get the scale factor from rec to w_params_num,
    e.g. scipy.misc.im
    (rec, size=scale_factor, interp='nearest')
    would make the rec into the same scale as w_params_num.
    """

    rec_pixel_size = get_pixel_size_um_rec(w_params_rec, setup_xscale)[0]
    stack_pixel_sizes = get_pixel_size_um_rec(w_params_stack, setup_xscale)[0]

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


def get_pixel_size_um_rec(w_params_num, setup_xscale):
    """Return the real length (in um) of each pixel point."""

    zoom = w_params_num.get('Zoom', w_params_num.get('zoom', np.nan))

    nxpix_all = w_params_num.get('User_dxPix', w_params_num.get('user_dxpix', np.nan))
    nxpix_retrace = w_params_num.get('User_nPixRetrace', w_params_num.get('user_npixretrace', np.nan))
    nxpix_lineoffset = w_params_num.get('User_nXPixLineOffs', w_params_num.get('user_nxpixlineoffs', np.nan))

    nxpix = nxpix_all - nxpix_retrace - nxpix_lineoffset

    assert nxpix > 0, w_params_num.keys()

    len_stack_x_um = setup_xscale / zoom
    stack_pixel_size_x = len_stack_x_um / nxpix
    stack_pixel_size_y = stack_pixel_size_x

    stack_pixel_size_z = w_params_num.get('ZStep_um', w_params_num.get('zstep_um', np.nan))

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


def get_weighted_linestack_xy(zidx, linestack, decay=5., power=2, plot=False):
    assert linestack.ndim == 3
    assert 0 <= zidx <= linestack.shape[2] - 1

    linestack = linestack
    zdists = np.concatenate([np.arange(zidx, 0, -1), np.arange(0, linestack.shape[2] - zidx)])
    zweights = np.exp(-np.abs(zdists / decay) ** power)
    weighted_linestack_xy = np.max(linestack * zweights, axis=2)

    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(12, 5))
        axs[0].plot(zdists)
        axs[1].plot(zweights)
        im = axs[2].imshow(linestack[:, :, zidx], cmap='Greys')
        plt.colorbar(im, ax=axs[2])
        im = axs[3].imshow(weighted_linestack_xy)
        plt.colorbar(im, ax=axs[3])
        plt.show()

    return weighted_linestack_xy


def match_template_to_image(image, template, original, exp_center,
                            dist_score_factor=1e-3, roi_coords_rot=None, plot=True, savefilename=None):
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

    tot_score = dist_score_factor * dist_score + match_score

    score = np.max(tot_score)
    y, x = np.unravel_index(np.argmax(tot_score), tot_score.shape)

    if plot:
        zoomextraw = 0.3 * np.max([htemplate, wtemplate])

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs = axs.flatten()

        ax = axs[0]
        ax.set_title('original')
        if original is not None:
            im = ax.imshow(original, cmap='rainbow', origin='lower')
            plt.colorbar(im, ax=ax, label='level')
            ax.set_xlim(axs[0].get_xlim())
        if roi_coords_rot is not None:
            for i, (roix, roiy) in enumerate(roi_coords_rot.T, start=1):
                ax.scatter(roix, roiy, marker='o', color='none', ec='k', s=150, alpha=0.8)
                ax.scatter(roix, roiy, marker='o', color='white', ec='none', s=150, alpha=0.4)
                ax.text(roix, roiy, i, color='k', ha='center', va='center')

        ax = axs[1]
        im = ax.imshow(template, cmap='rainbow', origin='lower', vmin=0, vmax=1)
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
            ax.plot([exp_center[0], x + wtemplate / 2], [exp_center[1], y + htemplate / 2], '-x', c='darkred',
                    zorder=100)

        ax = axs[3]
        im = ax.imshow(tot_score, cmap='rainbow', origin='lower', vmin=np.min(tot_score), vmax=np.max(tot_score),
                       extent=(
                           wtemplate // 2, tot_score.shape[1] + wtemplate // 2,
                           htemplate // 2, tot_score.shape[0] + htemplate // 2,
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
            ax.set_title(f'zoom - score={score:.3g}')
            ax.plot([x, x + wtemplate, x + wtemplate, x, x], [y, y, y + htemplate, y + htemplate, y], c='r')
            ax.set(xlim=(np.maximum(x - zoomextraw, 0), x + wtemplate + zoomextraw),
                   ylim=(np.maximum(y - zoomextraw, 0), y + htemplate + zoomextraw))

        ax = axs[-1]
        template = template.copy().astype('float')
        if np.any(~np.isfinite(template)):
            ax.contour(np.isfinite(template), levels=[0.5], zorder=10, colors=['pink'], linestyles=[':'],
                       extent=(x, x + wtemplate, y, y + htemplate))
        with np.errstate(invalid='ignore'):
            template[np.isfinite(template) & np.less(template, np.nanmean(template))] = np.nan
        ax.imshow(template, cmap='Greys', extent=(x, x + wtemplate, y, y + htemplate), alpha=0.5, vmin=-1, vmax=1,
                  origin='lower')
        if roi_coords_rot is not None:
            ax.scatter(x + roi_coords_rot[0, :], y + roi_coords_rot[1, :], marker='o', color='none', ec='k',
                       s=100, alpha=1)

        fig.suptitle('recording matching to morphology', y=1, va='bottom')
        plt.tight_layout()

        if savefilename is not None:
            fig.savefig(savefilename, facecolor='white')

        plt.show()

    return score, (x, y)


def find_best_match(
        linestack, d_rec_rot, exp_center=None, decay=5, power=2, blur=1, postblurpower=0.5,
        rescales=None, threshold=True, dist_score_factor=1e-3,
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
    best_linestack_xy = None
    best_xyz = None
    best_rescale = None
    best_d_rec_rot_fit = None
    best_d_rec_rot_rescaled = None

    if blur is not None:
        linestack = blur_linestack(linestack, blur=blur, postblurpower=postblurpower, inplace=False)

    if rescales is None:
        rescales = [1.0]

    for rescale in np.unique(rescales):
        d_rec_rot_rescaled = resize_image(d_rec_rot.copy(), rescale, dtype="float", order=0).T

        if threshold:
            d_rec_rot_fit = np.zeros(d_rec_rot_rescaled.shape, dtype='float32')
            d_rec_rot_fit[(np.nan_to_num(d_rec_rot_rescaled) >
                           np.nanmedian(d_rec_rot_rescaled) + 0.25 * np.nanstd(d_rec_rot_rescaled))
                          & np.isfinite(d_rec_rot_rescaled)] = 1.0
            d_rec_rot_fit[~np.isfinite(d_rec_rot_rescaled)] = np.nan
        else:
            lsmin = np.nanpercentile(d_rec_rot_rescaled, 20)
            lsmax = np.nanpercentile(d_rec_rot_rescaled, 80)

            d_rec_rot_fit = d_rec_rot_rescaled - lsmin

            if (lsmax - lsmin) > 0:
                d_rec_rot_fit = d_rec_rot_fit / (lsmax - lsmin)
            else:
                d_rec_rot_fit /= d_rec_rot_fit.max()

            d_rec_rot_fit = np.clip(d_rec_rot_fit, 0., 1.)

        for zidx in zidxs:

            weighted_linestack_xy = get_weighted_linestack_xy(
                zidx=zidx, linestack=linestack, decay=decay, power=power, plot=False).T

            score, pos = match_template_to_image(
                image=weighted_linestack_xy, template=d_rec_rot_fit, plot=plot_steps,
                original=d_rec_rot_rescaled, exp_center=exp_center, dist_score_factor=dist_score_factor)

            if best_score is None or score > best_score:
                best_score = score
                best_linestack_xy = weighted_linestack_xy
                best_xyz = pos[0], pos[1], zidx
                best_rescale = rescale
                best_d_rec_rot_fit = d_rec_rot_fit
                best_d_rec_rot_rescaled = d_rec_rot_rescaled

    assert match_template_to_image(
        image=best_linestack_xy, template=best_d_rec_rot_fit,
        original=best_d_rec_rot_rescaled, exp_center=exp_center,
        roi_coords_rot=resize_image(roi_coords_rot.copy(), best_rescale, dtype="float", order=0).T,
        plot=plot_result, savefilename=savefilename, dist_score_factor=dist_score_factor)[0] == best_score
    return best_xyz, best_rescale


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


def add_soma_to_linestack(linestack, pixel_sizes_stack, soma, radius_xyz, fill_value=0.15):
    linestack = linestack.copy().astype(float)

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
