"""Unit tests for djimaging.utils.motion on synthetic recordings with known motion."""
import numpy as np
import pytest
from scipy import ndimage

from djimaging.utils.motion import (
    clean_channels, compute_trust, compute_zproxy, estimate_motion, estimate_roi_shifts, extract_traces_weighted,
    make_reference, pool_roi_shifts, presentation_roi_weights, register_stack, roi_ids_from_mask, smooth_shift_field,
    trust_per_repeat, xcorr_disp)
from djimaging.utils.motion.image_utils import masked_pearson, pearson, reg_image
from djimaging.utils.motion.refine import pooled_mean_image, refine_roi_mask
from djimaging.utils.motion.registration import lowpass_disp
from djimaging.utils.motion.roi_shifts import local_shifts_batch, roi_centroids
from djimaging.utils.motion.trust import fast_per_bin, stim_bins

NX = NY = 64
FS = 7.8125
NPIX = 2


def _field(seed=0, n_cells=40):
    """Functional image (cells), structural image (vessels) and an Igor-format ROI mask of the cells."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    ch0 = np.full((NX, NY), 100.)
    mask = np.ones((NX, NY), int)
    centers = []
    while len(centers) < n_cells:
        c = rng.uniform(6, 58, 2)
        if all(np.hypot(*(c - d)) > 7 for d in centers):
            centers.append(c)
    for i, (cx, cy) in enumerate(centers):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        ch0 += rng.uniform(150, 400) * np.exp(-r2 / (2 * 2.2 ** 2))
        mask[(r2 <= 2.3 ** 2) & (mask == 1)] = -(i + 1)
    ch1 = np.full((NX, NY), 50.)
    for a, b, w in ((0.6, 5, 1.0), (-0.4, 50, 1.3), (1.8, -40, 0.8)):
        d = np.abs(y - (a * x + b)) / np.sqrt(1 + a ** 2)
        ch1 += 600 * np.exp(-d ** 2 / (2 * w ** 2))
    return ch0, ch1, mask


def _fourier_shift(img, shift):
    return np.real(np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(img), shift)))


def _recording(disp, n_frames=None, seed=1, blur=None, noise=8.):
    """Stacks (x, y, t) with displacement ``disp`` (n_frames, 2) and optional per-frame blur (defocus)."""
    ch0_img, ch1_img, mask = _field()
    n_frames = len(disp) if n_frames is None else n_frames
    rng = np.random.default_rng(seed)
    ch0 = np.empty((NX, NY, n_frames), np.float32)
    ch1 = np.empty((NX, NY, n_frames), np.float32)
    for i in range(n_frames):
        a, b = ch0_img, ch1_img
        if blur is not None and blur[i] > 0:
            a, b = ndimage.gaussian_filter(a, blur[i]), ndimage.gaussian_filter(b, blur[i])
        ch0[:, :, i] = _fourier_shift(a, disp[i]) + rng.normal(0, noise, (NX, NY))
        ch1[:, :, i] = _fourier_shift(b, disp[i]) + rng.normal(0, noise, (NX, NY))
    ch0[:NPIX] = 3000.
    ch1[:NPIX] = 3000.
    return ch0, ch1, mask


@pytest.fixture(scope='module')
def reference():
    ch0, ch1, mask = _recording(np.zeros((160, 2)))
    c0, c1, info = clean_channels(ch0, ch1, NPIX)
    return make_reference(c0, c1, info['npixartifact'], info['art_mask']), mask


def test_register_stack_matches_ndimage():
    rng = np.random.default_rng(0)
    stack = rng.normal(size=(32, 32, 20)).astype(np.float32)
    disp = rng.uniform(-3, 3, (20, 2))
    out = register_stack(stack, disp)
    for i in range(20):
        ref = ndimage.shift(stack[:, :, i], -disp[i], order=1, mode='nearest')
        ok = np.isfinite(out[:, :, i])
        assert np.allclose(out[:, :, i][ok], ref[ok], atol=1e-4)
        assert ok.mean() > 0.6


@pytest.mark.parametrize('shift', [(0.3, -0.7), (2.4, 1.1), (-3.6, 0.2)])
def test_xcorr_disp_recovers_subpixel_shift(shift):
    ch0, ch1, _ = _field()
    ref = reg_image(ch0, ch1, 0)
    img = reg_image(_fourier_shift(ch0, shift), _fourier_shift(ch1, shift), 0)
    disp, quality = xcorr_disp(ref, img)
    assert np.allclose(disp, shift, atol=0.1)
    assert quality > 5


def test_estimate_motion_separates_slow_drift_and_fast_jitter(reference):
    ref, _ = reference
    n = 400
    t = np.arange(n) / FS
    disp = np.stack([1.5 * t / t[-1], -0.8 * t / t[-1]], axis=1)
    jitter = (t > 20) & (t < 28)
    disp[jitter, 1] += np.sin(2 * np.pi * 1.0 * t[jitter])
    ch0, ch1, _ = _recording(disp, seed=2)
    c0, c1, info = clean_channels(ch0, ch1, NPIX)
    motion = estimate_motion(c0, c1, ref, info['npixartifact'], FS, art_mask=info['art_mask'])
    truth = disp[motion.bin_centers.astype(int)]
    calm = ~jitter[motion.bin_centers.astype(int)]
    assert np.sqrt(np.mean((motion.disp_bins[calm] - truth[calm]) ** 2)) < 0.1
    fast = np.linalg.norm(motion.fast_bins, axis=1)
    assert fast[~calm].max() > 0.5
    assert np.mean(fast[calm] > 0.5) < 0.05
    assert motion.slow_frames.shape == (n, 2)


def test_clean_channels_removes_bleedthrough_and_extends_artifact():
    n = 320
    rng = np.random.default_rng(3)
    ch0_img, ch1_img, mask = _field()
    activity = rng.normal(0, 1, n).cumsum()
    activity = (activity - activity.mean()) / activity.std()
    light = (np.sin(2 * np.pi * np.arange(n) / 40) > 0).astype(float)
    cells = (mask < 0).astype(float)
    ch0 = (ch0_img[:, :, None] * (1 + 0.3 * cells[:, :, None] * activity[None, None, :])
           + rng.normal(0, 5, (NX, NY, n))).astype(np.float32)
    ch1 = (ch1_img[:, :, None] + 0.1 * (ch0 - ch0.mean(axis=2, keepdims=True))
           + rng.normal(0, 5, (NX, NY, n))).astype(np.float32)
    ch1[:NPIX + 3] += 2000 * light[None, None, :]  # artifact wider than npixartifact
    ch0[:NPIX] = 3000.
    c0, c1, info = clean_channels(ch0, ch1, NPIX)
    assert info['npixartifact'] >= NPIX + 3
    assert 0.07 < info['k_bleed'] < 0.13
    # remaining activity in the structural channel is much weaker than before
    resid = c1[mask < 0] - c1[mask < 0].mean(axis=1, keepdims=True)
    before = ch1[mask < 0] - ch1[mask < 0].mean(axis=1, keepdims=True)
    assert np.std(resid) < 0.5 * np.std(before)
    assert np.all(c0[:info['npixartifact']] == c0[info['npixartifact']:info['npixartifact'] + 1])


def test_masked_pearson_matches_loop():
    rng = np.random.default_rng(4)
    X, Y = rng.normal(size=(5, 100)), rng.normal(size=100)
    W = (rng.uniform(size=(3, 100)) > 0.5).astype(float)
    r = masked_pearson(X, Y, W)
    for i in range(3):
        for b in range(5):
            sel = W[i] > 0
            assert np.isclose(r[i, b], pearson(X[b, sel], Y[sel]))


def test_local_shifts_batch_recovers_shift():
    ch0, _, _ = _field()
    patches_ref = np.stack([ch0[10:27, 10:27], ch0[30:47, 30:47]])
    shifted = _fourier_shift(ch0, (1.3, -0.6))
    patches_img = np.stack([shifted[10:27, 10:27], shifted[30:47, 30:47]])
    s = local_shifts_batch(patches_ref, patches_img, max_shift=2)
    assert np.allclose(s, [[1.3, -0.6], [1.3, -0.6]], atol=0.25)


def test_estimate_roi_shifts_and_weights_follow_local_displacement(reference):
    ref, mask = reference
    ids = roi_ids_from_mask(mask)
    shifted = _fourier_shift(ref['ch0'], (0.8, -0.5))
    shifts, quality = estimate_roi_shifts(shifted, ref['ch0'], mask)
    good = quality > 0.5
    assert good.mean() > 0.8
    assert np.allclose(np.median(shifts[good], axis=0), (0.8, -0.5), atol=0.15)
    weights, info = presentation_roi_weights(shifted, np.ones((NX, NY)), ref['ch0'], mask, shifts, NPIX)
    cw = np.array([ndimage.center_of_mass(w) for w in weights])
    c0 = roi_centroids(mask, ids)
    assert np.allclose(np.median(cw - c0, axis=0), (0.8, -0.5), atol=0.2)
    assert sum(v['low_contrast'] for v in info.values()) <= 2


def test_pool_roi_shifts_shrinks_noise_and_keeps_reproducible_shifts():
    rng = np.random.default_rng(5)
    cent = rng.uniform(0, 64, (60, 2))
    smooth = np.stack([0.02 * (cent[:, 0] - 32), 0.5 * np.exp(-((cent[:, 1] - 20) ** 2) / 200)], axis=1)
    noisy = [smooth + rng.normal(0, 0.1, smooth.shape) for _ in range(3)]
    quality = [np.ones(60)] * 3
    final, info = pool_roi_shifts(cent, noisy, quality)
    assert info['lam'] < 0.5
    assert np.sqrt(np.mean((final - smooth) ** 2)) < 0.07
    # reproducible individual displacements are kept (high lambda)
    indiv = smooth.copy()
    indiv[::5] += rng.choice([-1, 1], (12, 2)) * 0.8
    reps = [indiv + rng.normal(0, 0.05, indiv.shape) for _ in range(3)]
    final2, info2 = pool_roi_shifts(cent, reps, quality)
    assert info2['lam'] > 0.8
    assert np.sqrt(np.mean((final2 - indiv) ** 2)) < 0.1
    # single presentation: fixed lambda
    _, info3 = pool_roi_shifts(cent, reps[:1], quality[:1], lam_single=0.5)
    assert info3['lam'] == 0.5 and info3['n_presentations'] == 1
    assert smooth_shift_field(cent, smooth, sigma=np.inf).std(axis=0).max() < 1e-9


def test_zproxy_and_trust_detect_defocus(reference):
    ref, mask = reference
    n = 320
    blur = np.zeros(n)
    blur[160:240] = 2.5  # synthetic cells are smoother than real somata, so a stronger blur is needed
    ch0, ch1, _ = _recording(np.zeros((n, 2)), seed=6, blur=blur)
    c0, c1, info = clean_channels(ch0, ch1, NPIX)
    motion = estimate_motion(c0, c1, ref, info['npixartifact'], FS, art_mask=info['art_mask'])
    zp = compute_zproxy(c0, c1, ref, motion.disp_frames, info['npixartifact'], art_mask=info['art_mask'])
    edges = zp['bin_edges']
    mid = 0.5 * (edges[1:] + edges[:-1])
    defocus = (mid > 165) & (mid < 235)
    calm = (mid < 150) | (mid > 250)
    for k in ('self_ch0', 'self_ch1'):
        assert np.median(zp[k][calm]) > 0.95
        assert np.median(zp[k][defocus]) < np.median(zp[k][calm]) - 0.05
    reg0 = register_stack(c0, motion.slow_frames)
    fast = fast_per_bin(motion.fast_bins, motion.bin_centers, edges)
    res = compute_trust(reg0, ref, mask, edges, fast, info['npixartifact'], zself_ch1=zp['self_ch1'])
    assert np.median(res.trust[:, calm]) > 0.8
    assert np.median(res.trust[:, defocus]) < np.median(res.trust[:, calm]) - 0.2
    assert res.trust.shape == (len(roi_ids_from_mask(mask)), len(edges) - 1)


def test_trust_per_repeat():
    trust = np.ones((3, 10))
    trust[1, 4] = 0.2
    edges = np.arange(11) * 16
    triggertimes = np.array([0, 2, 10, 12]) * 16 / FS * 0.5
    rep, starts = trust_per_repeat(trust, edges, FS, triggertimes, ntrigger_rep=2)
    assert rep.shape == (3, 2) and len(starts) == 2
    assert rep[1].min() == pytest.approx(0.2)
    assert trust_per_repeat(trust, edges, FS, triggertimes, ntrigger_rep=0) == (None, None)


def test_extract_traces_weighted_ignores_nan():
    stack = np.ones((4, 4, 3), np.float32)
    stack[0, 0, :] = np.nan
    stack[1, 1, :] = 3.
    w = np.zeros((1, 4, 4))
    w[0, 0, 0] = 1.
    w[0, 1, 1] = 1.
    assert np.allclose(extract_traces_weighted(stack, w)[:, 0], 3.)


def test_full_chain_runs_on_tiny_synthetic_recordings(synthetic_recording_path):
    """Smoke test of the chain used by the motion tables on the 8x8 px tutorial recordings."""
    import h5py
    stacks = {}
    for stim in ('chirp', 'MB'):
        with h5py.File(synthetic_recording_path(stim), 'r') as f:
            stacks[stim] = (f['wDataCh0'][:].astype(np.float32), f['wDataCh1'][:].astype(np.float32),
                            f['ROIs'][:].astype(int))
    r0, r1, mask = stacks['chirp']
    c0, c1, info = clean_channels(r0, r1, 1)
    ref = make_reference(c0, c1, info['npixartifact'], info['art_mask'])
    p0, p1, pinfo = clean_channels(*stacks['MB'][:2], 1)
    motion = estimate_motion(p0, p1, ref, pinfo['npixartifact'], 10., art_mask=pinfo['art_mask'])
    zp = compute_zproxy(p0, p1, ref, motion.disp_frames, pinfo['npixartifact'], border=0)
    reg0 = register_stack(p0, motion.slow_frames)
    shifts, quality = estimate_roi_shifts(np.nanmean(reg0, axis=2), ref['ch0'], mask)
    final, _ = pool_roi_shifts(roi_centroids(mask), [shifts], [quality])
    weights, _ = presentation_roi_weights(np.nanmean(reg0, axis=2), np.mean(np.isfinite(reg0), axis=2), ref['ch0'],
                                          mask, final, pinfo['npixartifact'])
    edges = zp['bin_edges']
    res = compute_trust(reg0, ref, mask, edges, fast_per_bin(motion.fast_bins, motion.bin_centers, edges),
                        pinfo['npixartifact'], zself_ch1=zp['self_ch1'], roi_shifts=final)
    traces = extract_traces_weighted(reg0, weights)
    assert np.all(np.isfinite(motion.disp_bins))
    assert res.trust.shape[0] == len(roi_ids_from_mask(mask)) and np.all(np.isfinite(res.trust))
    assert traces.shape == (p0.shape[2], len(roi_ids_from_mask(mask))) and np.all(np.isfinite(traces))


def _dilate_rois(mask, grow):
    """Each ROI dilated (grow > 0) or eroded (grow < 0) by one pixel, without touching other ROIs."""
    out = np.ones_like(mask)
    for rid in roi_ids_from_mask(mask):
        m = mask == rid
        m = ndimage.binary_dilation(m) & ((mask == rid) | (mask == 1)) if grow > 0 else ndimage.binary_erosion(m)
        out[m & (out == 1)] = rid
    return out


def test_refine_roi_mask_trims_too_large_and_grows_too_small_rois():
    img, _, mask = _field()
    ids = roi_ids_from_mask(mask)
    for grow, sign in ((1, -1), (-1, 1)):
        drawn = _dilate_rois(mask, grow)
        refined, info = refine_roi_mask(img, drawn, NPIX)
        n_drawn = np.array([(drawn == i).sum() for i in ids])
        n_ref = np.array([(refined == i).sum() for i in ids])
        n_true = np.array([(mask == i).sum() for i in ids])
        # moves towards the cell: smaller if drawn too large, larger if drawn too small
        assert np.mean(np.sign(n_ref - n_drawn) == sign) > 0.8
        assert np.median(np.abs(n_ref - n_true)) < np.median(np.abs(n_drawn - n_true))
        assert set(info) == set(int(i) for i in ids)
        for i in ids:
            d = drawn == i
            assert not np.any((refined == i) & ~ndimage.binary_dilation(d))  # at most 1 px out
            assert not np.any(ndimage.binary_erosion(d) & (refined != i))  # the core is kept
            assert not np.any((refined == i) & (drawn < 0) & ~d)  # never takes pixels of other ROIs


def test_refine_roi_mask_keeps_dim_rois_and_respects_artifact():
    img, _, mask = _field()
    img = img + np.random.default_rng(3).normal(0, 3, img.shape) + 11000.  # ScanM-like offset
    mask = mask.copy()
    # an "ROI" without a cell next to the light artifact: background only
    dim = np.zeros_like(mask, bool)
    dim[NPIX:NPIX + 3, 30:33] = True
    mask[dim] = -99
    img[np.abs(np.arange(NX) - NPIX - 1)[:, None] + np.abs(np.arange(NY) - 31)[None, :] <= 6] =         np.median(img) + np.random.default_rng(4).normal(0, 3, 1)[0]
    refined, info = refine_roi_mask(img, mask, NPIX)
    assert info[-99]['status'] == 'low_contrast'
    assert np.array_equal(refined == -99, mask == -99)
    assert not np.any(refined[:NPIX] < 0)  # never grows into the light artifact
    assert set(np.unique(refined[refined < 0])) == set(np.unique(mask[mask < 0]))
    assert sum(d['status'] == 'low_contrast' for d in info.values()) == 1


def test_pooled_mean_image_ignores_nan():
    a = np.full((4, 4), 1.)
    b = np.full((4, 4), 3.)
    b[0, 0] = np.nan
    p = pooled_mean_image([a, b], [1, 3])
    assert p[0, 0] == 1. and p[1, 1] == 2.5


def test_stim_bins_excludes_pre_stimulus():
    starts = np.arange(10) * 16
    sel, onset = stim_bins(starts, FS, np.array([40 / FS, 100 / FS]))
    assert onset == pytest.approx(40 / FS)
    assert np.array_equal(sel, starts >= 40)
    sel, onset = stim_bins(starts, FS, np.array([]))
    assert onset is None and sel.all()
    sel, _ = stim_bins(starts, FS, np.array([1000.]))  # no bin after onset: fall back to all bins
    assert sel.all()


def test_pick_frame_bins_and_plot_motion_frames():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from djimaging.utils.motion.plotting import (
        bins_at_times, mark_frames, mark_stimulus, pick_frame_bins, plot_motion_frames, roi_outline_segments)
    score = np.zeros(20)
    score[[7, 8, 15]] = [3., 2.5, 1.]
    idxs, worst = pick_frame_bins(score, n_spread=3, n_worst=2)
    assert {0, 10, 19} <= set(idxs) and 7 in idxs and 15 in idxs and 8 not in idxs  # 8 too close to 7
    assert set(idxs[worst]) == {7, 15}
    edges = np.arange(21) * 4
    assert np.array_equal(bins_at_times([0., 4.5 / FS, 1e3], edges, FS), [0, 1, 19])

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(16, 12, 80)).astype(np.float32)
    mask = np.zeros((16, 12), int)
    mask[3:6, 3:6] = -1
    mask[8:11, 5:9] = -2
    fig = plt.figure(layout='constrained')
    sf = fig.subfigures(2, 1)
    labels = plot_motion_frames(sf[1], raw, raw, raw.mean(axis=2), edges, idxs, FS, roi_mask=mask, is_worst=worst,
                                stim_onset=10., disp=np.zeros((len(idxs), 2)), gamma=0.6)
    assert labels == list('ABCDE')
    segs = roi_outline_segments(mask)
    assert len(segs) == 2 * (3 + 3) + 2 * (3 + 4)  # pixel edges around the two rectangular ROIs only (not background)
    idxs_c, worst_c = pick_frame_bins(score, n_spread=3, n_worst=2, candidates=np.arange(20) >= 10)
    assert set(idxs_c[worst_c]) == {15, 12}  # 7 (highest score) is before the candidate range
    ax = sf[0].subplots()
    mark_stimulus(ax, [2., 8.], label=True)
    mark_frames(ax, idxs, edges, FS, labels, worst)
    fig.canvas.draw()
    plt.close(fig)


def test_lowpass_disp_split_keeps_step_at_onset():
    dt = 4 / FS
    disp = np.zeros((200, 2))
    disp[40:, 0] = 2.  # focus / position adjusted just before the stimulus starts at bin 40
    fast_joint = np.abs(disp - lowpass_disp(disp, dt, 10.))[40:60, 0]
    fast_split = np.abs(disp - lowpass_disp(disp, dt, 10., breaks=[40]))[40:, 0]
    assert fast_joint.max() > 0.5
    assert fast_split.max() < 1e-6
    assert np.array_equal(lowpass_disp(disp, dt, 10., breaks=[0, 200]), lowpass_disp(disp, dt, 10.))
