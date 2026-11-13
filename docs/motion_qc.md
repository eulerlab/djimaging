# Motion correction and motion QC ("motion mode")

Method, design decisions and validation of the motion tables.

> ⚠️ **Experimental.** Motion detection, the presentation ROI masks, the trust score and the HighRes alignment were
> validated on a limited set of recordings: the xy RGC fields of four experiments in `imaging_2p_data` (2018 and 2022,
> setups 1 and 3) and one z-stack. Thresholds (trust, z) and the conversion of similarities to µm are not calibrated
> beyond these data. Check the plots (`plot1`) of your data before excluding ROIs or repeats based on them.

- Implementation: `djimaging/utils/motion/` (computation) and `djimaging/tables/motion_correction/` (tables
  `MotionParams`, `MotionReference`, `MotionDetection`, `MotionRoiMask`, `MotionTrust`).
- Use: `python make_project.py <project> --blocks motion,...` switches a schema into motion mode: `RoiMask` only stores
  the field mask (`_presentation_masks = False`, no `RoiMaskPresentation`), `MotionRoiMask` provides the presentation
  ROI masks, and `Traces` (`_include_motion_correction = True`) are extracted from the slowly registered stack with
  these masks. See `djimaging/schemas/blocks/motion.py` and `djimaging/notebooks/blocks/motion.py`.
- Validation was done with prototype scripts outside the repository; the numbers below refer to the test data in
  `imaging_2p_data` (xy-RGCs, zstack). Parameter names correspond to `DEFAULT_MOTION_PARAMS`.

Date of the validation: 2026-11-07.

## Problem

ROI masks are drawn once per field (on the first stimulus of C1) and, if at all, shifted globally by a few pixels
for other stimuli / conditions. Slow drift between stimuli is often corrected manually by experimenters. We want:

1. To know *when* the signal can be trusted: per ROI and per recording, as a function of time.
2. ROI masks that may differ slightly between stimuli / conditions (e.g. after a slight z shift).
3. To handle slow drift *within* a recording.

Good recordings are already stable; the goal is to get the most out of slightly drifting recordings.

## Data

- 64x64 px, ~7.8 Hz, OGB-1 bulk-loaded wholemount retina, GCL.
- `wDataCh0`: OGB-1 (functional). `wDataCh1`: SR101 (blood vessels, some filled cells; activity-independent).
  `wDataCh2`: trigger.
- Test data: `imaging_2p_data/xy-RGCs/20220215/1/Pre/SMP_M1_LR_{field}_{stim}_{cond}.h5`, fields GCL0–3,
  stimuli chirp / DN / MB, conditions C1 (first), then C2 (control again) or D (drug) with the same stimulus order.
- Ratiometric OGB/SR101 correction is **not** applicable: almost no ROI has its own SR101 signal.

## Decisions (from Q&A)

| Topic | Decision |
|---|---|
| Output | QC always. Corrected traces optional; they *replace* the default traces (never store both). |
| Phase | Prototype script/notebook first, tables later. |
| z | No z-stacks (not planned) → relative z proxy only (residual similarity / shape change after xy correction). |
| Trust resolution | Fixed time bins + aggregated per stimulus repeat. |
| Masks | Global shift per presentation + mild reshape (e.g. dropping border pixels); ROI identity kept. |
| ROI set | Fixed per field; ROIs that become unreliable are flagged, not removed. |
| Start masks | Igor `ROIs` stored in each h5. |
| QC basis | Structure only: SR101 (Ch1) + spatial shape / registration metrics. Never response amplitude or Ch0 brightness level (drug confound). |
| Test set | All GCL0–3. Then pick the most stable recording and simulate drift for a controlled test. Real drifting example may follow. |
| Registration | Simple scipy/skimage phase correlation baseline; compare with a heavier package (suite2p). |
| Within-recording drift | Correct only smoothed (slow) shifts. QC must clearly flag when the field moves faster than what is corrected. |
| Reference | First stimulus of C1 (stable-period average). |

## Approach

1. **Motion estimation per recording.** Rigid subpixel xy shifts (phase correlation, upsampled) on band-pass
   filtered, temporally binned frames, from Ch1 + Ch0 combined (Ch0 *structure* only, after band-pass; vessel
   channel alone has few xy landmarks). Reference = stable-period average of the field reference recording,
   iteratively refined.
2. **Slow vs. fast motion.** Low-pass the shifts → slow shifts (corrected). Residual |raw − slow| → fast motion
   (not corrected) → QC flag if it exceeds a fraction of a pixel / ROI size.
3. **Z proxy.** After applying xy shifts, correlation of each time bin to the reference (band-passed, per channel).
   Drops not explained by xy → shape / focus change ≈ z. Ch1 vessels are thin, hence z-sensitive.
4. **Per-ROI trust over time.** For each ROI and time bin: local patch correlation to the reference patch after
   global xy correction (Ch0 band-passed + Ch1), local residual shift of the patch, and mask coverage (fraction
   of the shifted mask inside the FOV). Combined into a trust score in [0, 1]; aggregated per repeat and per
   recording.
5. **Per-presentation masks.** Global shift of the field mask (from the presentation's mean image vs. reference),
   then mild reshape per ROI: drop pixels that leave the FOV or no longer match (low local correlation / outside
   the cell in the new mean image). ROI ids fixed; flag ROIs whose mask lost too much / whose patch decorrelates.
6. **Controlled test.** Take the most stable recording, simulate (a) slow xy drift, (b) fast xy jitter, (c) z-like
   blur / defocus change of a period, (d) a whole-field offset between presentations. Check each is recovered /
   flagged.

## Outputs of the prototype

- Figures per field: shifts (raw / slow / fast), z proxy, per-ROI trust heatmap (ROI × time bin), per-repeat
  trust, mask comparison C1 vs. C2/D.
- Comparison of simple vs. suite2p shifts.
- A short results summary appended below.

## Prototype scripts (not in the repository)

- `motion_qc.py` — library: loading, registration (`xcorr_disp`), reference, motion, z proxy, per-ROI trust,
  per-presentation masks, trace extraction.
- `run_all.py` — all fields → `out/<field>_{overview,trust,masks}.png`, `out/<field>_results.pkl`, `out/summary.csv`.
- `simulate.py` — controlled drift test → `out/sim_<rec>.png`.
- `export_for_s2p.py`, `s2p_compare.py` (runs in a separate suite2p venv), `compare_s2p.py` → `out/s2p_comparison.*`.
- Run: `PYTHONPATH=<repo> .venv/Scripts/python run_all.py [GCL0 ...] [--jobs 4]` from this folder (~50 s for 4 fields / 24 recordings with 4 jobs; ~5 s per recording single-threaded).

### Method as implemented

- **Preprocessing (QC copy of the data, at load)** — needed so that Ch1 really is activity- and stimulus-independent:
  1. Light-artifact width: columns next to the `npixartifact` lines whose Ch1 time course follows the artifact lines
     are added to the exclusion zone (2018 setup-3 data: 4–6 instead of 3 lines, stimulus dependent).
  2. OGB bleed-through into Ch1: Ch1 −= k·(Ch0 − mean_t Ch0), one global k (k ≈ 0.04–0.14).
  3. Remaining Ch1 pixels correlated with the artifact are filled and excluded.
  4. Artifact columns of both channels are replaced by the first valid column — otherwise the band-pass (σ 4 px)
     smears the bright stripe ~10 px into the image and the structure "changes" with the stimulus.
  All correlations on 2 s bins, high-passed (10 s) so slow z drift is not mistaken for artifact.
  Without this, 2018 recordings showed stimulus-locked Ch1 "z" dips and false trust drops (GCL4_DS: trust 0.39 → 0.99).
- **Registration image**: band-passed (DoG 0.75/4 px) Ch0 and Ch1, each z-scored, summed, Tukey-tapered.
- **Shift estimate**: FFT cross-correlation with partial whitening (exponent 0.5), Gaussian smoothing of the
  correlation surface (1 px), search window ±8 px, subpixel via upsampled DFT (×20); 4-frame bins (~0.5 s).
  Full phase correlation (skimage default) failed under defocus (jumps of 20 px); this version does not.
- **Reference**: first C1 recording of the field (by timestamp = MB), iterative register-and-average, then all
  frames registered and averaged; even/odd-frame half means kept for noise correction. Coordinate frame = raw
  recording (median displacement 0), so Igor masks drawn on the raw image fit.
- **slow** = median-filter + Gaussian (σ = 10 s) of the displacement → applied to the stack before trace extraction.
  **fast** = raw − slow → not corrected, flagged (|fast| > 0.5 px; enters trust). Before and after the first trigger
  are smoothed separately (`split_at_onset`), so a focus / position adjustment before the stimulus is corrected as a
  step instead of being smeared into the stimulus period. Summary values (`MotionDetection` slow range / fast motion,
  `MotionTrust` median trust / flagged ROIs) only use bins starting after the first trigger.
- **z proxy** (2 s bins): Pearson corr of band-passed images after xy registration, *noise-corrected*
  (corr / sqrt(R_bin·R_target), R from split-half Spearman-Brown). Against the recording's own mean (within-recording
  change) and against the field reference (plane vs. reference); Ch1 = activity-independent.
- **Within-recording trust** per ROI and 2 s bin, aggregated per repeat (min over the repeat). Since 2026-11-13 the
  fraction of the ROI on its cell, with the z change reported separately (see "Trust, revised" below). Before:
  `soft(corr_self; 0.5..0.85) · gauss(res_shift; 0.75 px) · gauss(fast; 0.75 px) · soft(Ch1 z self; 0.75..0.95) · coverage`
  - corr_self: noise-corrected patch corr (Ch0, band-passed, ROI dilated by 2 px) vs. the recording's own mean.
  - res_shift: local displacement of the ROI patch vs. the **field reference** after slow correction (position only).
- **Across-presentation match** (one value per ROI): noise-corrected patch corr and local shift of the recording mean
  vs. the field reference. Reported, not folded into trust (confounded by drug / dye brightness changes).
- **Per-presentation mask**: field mask, integer global shift (current storage format) + mild reshape: drop pixels out
  of FOV, drop *border* pixels that were cell in the reference and are not anymore — only if the ROI kept ≥ 50 % of
  its contrast; otherwise flag the ROI as `low_contrast` instead of reshaping.

## Results (2026-11-07, GCL0–3, 24 recordings)

**Simulation** (GCL2_chirp_C1, most stable recording; known drift added with Fourier shifts on both channels:
offset (2.3, −1.4) px + linear ramp (2, −1) px, 15 s of 1 Hz / 1 px jitter, 40 s defocus ramp up to σ = 1.5 px):

| | result |
|---|---|
| shift error (RMSE) | 0.06 px calm, 0.09 px during defocus, 0.29 px during jitter (fast part not corrected, by design) |
| fast-motion flag | 73 % of jitter bins, 0 % false positives outside |
| global offset | (3, −2) recovered, true median (3.34, −1.93) |
| z proxy at σ = 1.5 px defocus | Ch1 self 0.91, Ch0 self 0.95 (calm ≥ 0.99) |
| trust (median ROI) | corrected 0.98 calm / 0.70 jitter / 0.67 defocus; global integer shift 0.54; fixed mask 0.00 |
| **trace fidelity** vs. unperturbed (median r) | **corrected 0.985**, global integer shift 0.76, fixed mask (current practice) 0.37 |

Trust ranks the three strategies like the actual trace fidelity.

**Real data** (`out/summary.csv`):
- Within-recording xy drift is small everywhere (slow range ≤ 0.9 px, mostly < 0.3 px); fast motion is rare
  (one real event: GCL0_DN_C1 at ~300 s, ~0.7 px, flagged; trust dips to ~0.4).
- Between conditions: global offsets up to ~1.6 px (GCL0 C2 x ≈ 1.4, GCL3 D y ≈ −1.6).
- **z is the main issue, not xy.** Ch1 (vessels) vs. field reference drops to 0.5–0.8 in most C2 / D recordings
  (C1: ≥ 0.86). GCL1_DN_D drifts *through* the reference plane during the recording (Ch1 vs. ref 0.55 → 0.95 → drop
  at ~300 s); Ch1 self-similarity 0.73 there, i.e. a stronger change than the simulated 1.5 px defocus. Trust there
  median 0.61, 15 / 90 ROIs flagged.
- **Non-rigid deformation between conditions**: after global correction, per-ROI residual shifts vs. the field
  reference are ~0.1 px in C1 (noise floor) but spatially structured in GCL3 D (lower-right region ~1–1.5 px, left half
  ~0), identical across MB/chirp/DN → a stable deformation (or tilt + z). ~30 % of ROIs > 0.75 px.
  GCL1 D shows a weaker, also consistent pattern (`out/roi_residual_shifts.png`). The mean residual (~0.3 px) also shows
  that the global shift (Ch0+Ch1, vessel-weighted) is not exactly the cells' shift when the plane changes.
- Per-presentation masks: ≥ 95 % of mask pixels kept; 17–63 ROIs per C2/D recording flagged `low_contrast`
  (0–11 in C1).
- **suite2p (1.1.0) rigid** registration is integer-pixel only (subpixel only in non-rigid blocks). On the same
  references and bins it agrees with our rounded shifts (median agreement 100 %; lower only where shifts sit near
  ±0.5 px), but cannot resolve the sub-pixel drift that matters here. No reason to add the dependency.

### All xy-RGCs experiments (72 recordings, 16 fields; `out/summary_all.csv`, figures in `out/<exp>/`)

Run: `run_all.py --jobs 6 --data <exp>/Pre ...` → 165 s for all.

| experiment | design | within-recording | between presentations |
|---|---|---|---|
| 20181011_1 (setup 3) | DS → Chirp → MC per field, one condition | very stable: trust med ≥ 0.94, Ch1 z ≥ 0.95, slow drift ≤ 1 px (GCL3_MC5, 10 min) | match ≥ 0.97, Ch1 vs ref ≥ 0.90 |
| 20181011_2 (setup 3) | same; GCL3 has a 2nd chirp ~17 min later | stable: trust med ≥ 0.91; Ch1 z ≥ 0.87 (GCL2_MC16) | match ≥ 0.98; GCL3_Chirp2 vs ref 0.98 → no drift over 17 min |
| 20220125_2 (setup 1) | MB → chirp → CSL, C1 then C2/D | very stable (fast motion ~0, slow ≤ 0.25 px) | offsets ≤ 1.2 px; z change in D for GCL1/GCL2 (Ch1 vs ref 0.72–0.79); GCL0 C2: Ch0 match 0.86–0.95, 12 ROIs flagged in CSL_C2 |
| 20220215_1 (setup 1) | MB → chirp → DN, C1 then C2/D | stable except GCL1_DN_D (z drift through the plane, trust 0.65) and fast event in GCL0_DN_C1 | z change in most C2/D; non-rigid deformation in GCL3 D (31–34 ROIs flagged) |

- Remaining low-trust bins in otherwise stable 2018 recordings (~5 %) are (a) ROIs drawn inside the stimulus-dependent
  light-artifact zone (coverage < 1; correctly flagged — those ROIs overlap the artifact) and (b) a few ROIs with local
  residual shift > 0.75 px (2018_2 GCL3_Chirp: 5 ROIs).
- Fast motion (> 0.5 px) is essentially absent in all 72 recordings (max 0.8 % of bins).
- Pattern across experiments: **within a condition, drift is small (xy < 1 px, z stable); the critical step is the
  break between C1 and C2/D** (xy offsets up to 1.6 px, z changes, non-rigid deformation).

## Q: remove ROI border pixels against subpixel drift? (`test_mask_erosion.py`)

No — move the mask with the tissue (registration / subpixel-shifted mask weights) and flag what cannot be corrected.
- ROIs are tiny: median 17 px (equivalent radius 2.3 px). A 1-px erosion keeps only 29 % of the pixels (2 ROIs vanish).
- Without correction, "erode until the ROI stays inside the cell" needs ceil(drift range) = 2 px in the simulation →
  78 / 112 ROIs vanish; the rest have *lower* fidelity (0.67) than full mask + global shift (0.74).
- With correction: full mask 0.986, 1-px eroded 0.980 (and noisier), soft mask (weights = fraction of time inside the
  ROI under the uncorrected residual motion) 0.986 → no gain.
- Fast jitter within a 2 s bin breaks every mask strategy equally (r ≈ 0.38) → flag it.
- (Separate topic: border pixels carry neighbour / neuropil signal even without drift → core-weighted masks could
  improve purity, but that is a contamination question, not drift.)

## Between-presentation local correction (`between_presentations.py`, `sim_between.py`)

Per-ROI displacement of each presentation's (slow-registered) mean vs. the field reference after global correction
(`estimate_roi_shifts`: local patch cross-correlation, ±3 px, subpixel; quality = peak corr).

**Validation on real data** (median per-ROI error vs. the held-out measured shifts; C1 ≈ noise floor 0.08–0.13 px):

| predictor | test | C2 | D |
|---|---|---|---|
| global only (current) | – | 0.39 px (13.6 % > 0.75 px) | 0.34 px (5.6 %) |
| robust mean of ROIs (better global) | cross-presentation | 0.33 | 0.33 |
| affine field | cross-presentation | 0.27 | 0.27 |
| Gaussian field σ = 3 / 6 px | cross-presentation | 0.13 / 0.22 | 0.12 / 0.18 |
| per-ROI shift of one other presentation | cross-presentation | 0.10 | 0.09 |
| **per-ROI shift pooled over the other presentations** | leave-one-presentation-out | **0.08** | **0.075** |
| pooled + EB shrinkage to σ = 4 field | leave-one-presentation-out | 0.12 | 0.10 |

→ Per-ROI displacements between conditions are real and reproducible across presentations (not noise), beyond a
smooth field (in GCL3 D partly a smooth deformation, partly individual ROIs).

**Simulation** (stable recording): (A) brightness change of 30 % of cells ×0.5–2 without motion (drug-like) does not
create spurious shifts (per-ROI median 0.12 px = noise, none > 0.5 px). (B) known smooth deformation (2 % scaling,
1° rotation, 1.2 px bump; |d| up to 1.5 px): recovered to 0.09 px (σ 6 field) / 0.18 px (raw per-ROI); trace fidelity
p10 0.84 (global) → 0.94 (local), median 0.979 → 0.993. (C) A+B: same.

**Implemented** (`condition_roi_shifts`, `refine_presentation_mask(..., roi_shifts)`, two-pass `run_all.py`):
per condition (block of presentations after a break; the reference recording itself has shift 0)
1. per-ROI shifts of all presentations of the condition → median per ROI;
2. robust smooth field (Gaussian kernel regression, σ = 4 px) on the pooled shifts;
3. per-ROI residual shrunk towards the field with empirical-Bayes λ = 1 − noise_var / total_var (noise from the
   disagreement between presentations; λ = 0.5 for a single presentation; low-quality ROIs → field);
4. applied as subpixel-shifted (bilinear) mask weights, then the mild border reshape; trust measures position residuals
   relative to the applied shift.
Flags per ROI: shift at the search limit (±3 px; likely mismatched cell), inconsistent across presentations (> 0.5 px).

**Effect** (all experiments, C2/D recordings; note: trust is evaluated on the same data the shifts were estimated
from → optimistic; the cross-validated errors above are the honest numbers):
- median trust 0.71–0.95 → 0.97–1.00; ROIs flagged (median trust < 0.5) e.g. 20220215 GCL3 D 31–34 → 5,
  GCL0 C2 6–7 → 0, GCL1 MB_D 21 → 1. C1 non-reference: 0.982 → 0.992 (no harm).
- λ is high (0.65–0.96) where displacements are large and reproducible (20220215 GCL0 C2 / GCL3 D, 20220125 GCL1/2 D)
  and low (0–0.5) in C1 (noise only → smooth field ≈ 0).
- Not correctable and still flagged: GCL1_DN_D (z drift within the recording; trust 0.82, 31 % low bins).
- The `low_contrast` flag fires for ~50 % of ROIs in some D recordings (drug changes OGB brightness) → needs
  recalibration / should not be read as "ROI lost" in D.

## Field ROI mask: which presentation, aligned (`RoiMask`, motion mode)

Igor masks are drawn on one presentation and copied (same or integer-shifted) to the others; which one is not stored.
`mask_alias` order picked Chirp in all 2018 fields, but only the DS files have a `Stack_ave` (written by the drawing)
and the mask fits DS best. Copied shifts are often wrong outside the drawing condition (20220215 GCL0 C2: copied
(-1, 0), best fit (2, -1)). In motion mode the presentation of the field mask is the motion reference, so `RoiMask`
now chooses it from the data (`mask_utils.select_field_roi_mask`):

- Fit = median over ROIs of (mean inside − mean in a 2 px background ring) of the high-passed (σ = 6 px), robustly
  z-scored mean image of the ROI channel after stimulus onset. Pixel noise is negligible in mean images (≤ 0.04 % of
  the variance even for 40 frames), so differences between presentations are real image differences (activity, z).
- Candidates: presentations whose mask is a copy of the `mask_alias` mask; each copy is aligned to its own image
  (integer offset ≤ 2 px, accepted only if the fit improves by > 10 %). Best fit wins; within 5 % a presentation with
  `Stack_ave` wins, then the `mask_alias` one. No positive fit → `mask_alias` (`mask_selection='alias'`, warning).
- Stored: `align_dx/dy` (the file keeps the mask as drawn; `check_filesystem_consistency` applies the offset),
  `mask_fit`, `mask_selection`.
- All 16 xy fields: offset (0, 0) everywhere; choice DS (7 of 8 fields of 2018, one MC5 with fit 0.33 vs. DS 0.25),
  `MB_C1` (with `Stack_ave`) or `chirp_C1` in 2022. C2/D fit 2–4× worse than C1, consistent with the z changes above.
- Files: in motion mode a field has one **field mask file** `<stem>_FieldROIs.npy` (`to_field_roi_mask_file`), named
  after the presentation it belongs to; no per-presentation copies are written (a copy for a shifted presentation would
  be wrong in a schema without motion mode, which never loads field mask files). Loading prefers a field mask file:
  its presentation is used without choice by fit or alignment (`mask_selection='file'`, `mask_fit` kept for QC);
  several field mask files of one field are an error. Written by the ROI canvas ('Save all to file' / deploy),
  `correct_roi_masks` and `RoiMask().export_field_roi_masks(roi_mask_dir='PreparedROIs')`, which by default also lets
  the entry reference the file with `align = (0, 0)` (mask unchanged, nothing downstream deleted). The per-presentation
  H5 masks (copies) are only the input of the selection; presentation masks in motion mode come only from
  `MotionRoiMask` (`Traces`, `rf_offset` and `RoiMask.plot1` follow this).
- ROI canvas: drawn and edited on the presentation of the mask only; the other presentations are view only and show
  the `MotionRoiMask` masks (`draw_roi_mask(motion_roi_mask_table=MotionRoiMask)`, while the mask is unedited) or a
  preview (mask moved by the auto shift between the mean images).

## Refined field ROI mask (`RefinedRoiMask`, optional; `refine_masks.py`, `out/refine/`)

Tested, no gain → optional block `motion_refine` in `make_project.py`, off by default. Without it, `MotionRoiMask` and
`MotionTrust` start from the drawn (aligned) `RoiMask` mask.

Method (`utils.motion.refine.refine_roi_mask`): on the registered mean of all presentations of the reference condition,
each ROI keeps its core (eroded by 1 px); border pixels within ±`refine_max_px` = 1 px are kept/added if
(img − local background) / (peak − background) ≥ `refine_rel_thresh`. ROI ids kept; never takes pixels from other ROIs
or the light artifact; dim ROIs (`refine_min_contrast`) and ROIs that would keep < `refine_min_keep_frac` stay as drawn.
Contrast must ignore the raw offset (~11000): relative to the raw background it is ~0.004 for every ROI.

All xy experiments (16 fields, 72 presentations, ~1300 ROIs), drawn vs. refined mask; QI across repeats (QI ≈ 0.5) and
correlation of the ROI trace with the surrounding neuropil:

| setting | ROIs changed | median size | Δ QI | Δ neuropil corr. |
|---|---|---|---|---|
| threshold 0.5, grow + trim (default) | 75 % | ×1.00 | −0.002 | ±0.000 |
| threshold 0.5, trim only | 70 % | ×0.86 | −0.003 | −0.004 |
| threshold 0.7, trim only | 70 % | ×0.72 | −0.008 | −0.010 |

- All changes < ~1.5 %, and a plain trade-off: growing raises QI slightly, trimming lowers the neuropil correlation
  slightly; no setting improves both. ROIs whose shape changed by > 50 % of their pixels gain nothing either (QI +0.002).
- Consistent with the erosion test above: at 1.5 µm/px ROIs have ~17 px, so ±1 px changes the shape a lot but the
  averaged trace hardly.
- The `low_contrast` flag of `MotionRoiMask` is not caused by the offset (it compares two contrasts against nearly the
  same background, so the offset cancels).

## Shrink to fit: border pixels that do not fit the ROI (`shrink_rois_to_fit`; `shrink_masks.py`, `shrink_synthetic.py`)

For masks drawn a little too large (by hand or AutoROIs). Not motion specific; ROI canvas ('Shrink ROI', 'Shrink all
ROIs') and `correct_roi_masks(shrink=True)` (files, review, deploy). Per ROI, on one recording: high-passed pixel traces;
`corr_in` = correlation with the rest of the ROI, `corr_out` = with the pixels outside within 2 px; reference = median
`corr_in` of the core. A border pixel does not fit if `corr_out > corr_in + margin` (belongs to the surround, e.g. a
neighbour) or `corr_in < min_rel_corr * reference` (no signal of the ROI, e.g. a dark gap / vessel); only connected
groups of `min_group` such pixels are removed; ROIs that would lose > 40 % or split stay as drawn.

Key observation (real data, pixel classes): correlation with the cell falls off smoothly outside the ROI (blur, somata
larger than the ROI): pixels 1 px outside keep a median of 0.51 of the ROI's correlation, 2 px outside 0.24-0.28. So a
pixel 1 px outside a cell is usually *not* decorrelated, and removing it hardly matters. Temporal binning raises the
correlations of wrong and correct pixels alike (no better separation).

Synthetic ground truth (cells r 2.3 px, PSF 0.6 px, own + shared activity, neuropil, noise matched to the real pixel
SNR; trace fidelity = corr with the true activity):

| ROI drawn 1 px too large into | fidelity true / drawn | strict default | sensitive preset |
|---|---|---|---|
| neuropil | 0.738 / 0.703 | not detected | not detected |
| touching neighbour (30 % shared activity) | 0.778 / 0.759 | not detected | 0-3 of 3 px (mixed pixels) |
| neighbour with own activity (unit test) | - | not detected | 14 of 15 px |
| dark gap / vessel | 0.697 / 0.696 | 16 % of px | 73-77 % of px |

No correct ROI was changed in any synthetic setting.

Real data (16 xy fields, 1340 ROIs), decided on the field mask presentation, evaluated on the other presentations:

| preset | parameters | ROIs changed | fields unchanged | dQI (same / other condition) | dNP |
|---|---|---|---|---|---|
| strict (default) | margin 0.3, min_rel_corr 0.0, min_group 2 | 23 (1.7 %), 0-4 per field | 6 / 16 | +0.003 (68 % > 0) / +0.001 | -0.001 |
| sensitive (`SHRINK_SENSITIVE`) | margin 0.15, min_rel_corr 0.2, min_group 1 | 210 (15.7 %) | 0 / 16 | -0.000 / -0.000 | -0.002 / -0.003 |

The strict default matches the expectation "most fields need nothing, some 1-3 ROIs"; the few ROIs it changes get
slightly better traces. The sensitive preset is for a single ROI that looks wrong in the canvas (undo restores it).

## ROIs without pixels in a presentation (ChirpQI failure, 2026-11-13)

`ChirpQI` failed with `Field 'qidx' doesn't have a default value`: `qidx` was NaN, and DataJoint omits NaN in a NOT NULL
column. Cause: in `ageuler_all_gcl_v2`, 1920 of 517658 ROI × presentation entries (0.37 %, 298 field/conditions from
2013 to 2024; median 9 px) have no pixel left in their `MotionRoiMask` weights (`n_px = 0`). Their motion trace was a
constant 0 (weighted mean without weight), which standardization (std 0) turns into NaN, and `qidx = 0/0`. Not single
trials, not NaN in the raw data, and not the NULL change of `MotionTrust`.
Why ROIs lose all pixels: ≥ 50 % touch the field edge (`dist_edge = 0`; pixels outside the FOV in > 5 % of the frames
are dropped), 34 % have a per-ROI shift at the search limit (±3 px; see next section).

Fixed:
- `extract_traces_weighted`: frames without any valid pixel of the ROI (and ROIs without weight) are NaN, not 0.
- `roi2trace_from_stack_motion`: such ROIs get `trace_valid = 0`; `PreprocessTraces` (and Cascade) only use valid
  traces, so nothing downstream breaks. The trace stays in `Traces` (flagged).
- `Traces` in motion mode: `incl_artifact` is decided on the field ROI mask (as without motion mode). The presentation
  label mask never contains artifact pixels (they are dropped from the weights), so with `include_artifacts = False`
  ROIs in the artifact were not skipped before.
- `RepeatQITemplate`: a non-finite `qidx` raises an error that names the key and the reason.

## Per-ROI shifts at the search limit (`pool_roi_shifts`, 2026-11-13)

A per-ROI shift at the search limit (`roi_max_shift`, ±3 px) is not a measurement: the patch most likely matched a
neighbouring cell. Before, it entered the pooled shift and the smooth field and moved small ROIs off their cell or out of
the FOV. Now such shifts are left out of the pooling and of the field fit; an ROI without any valid shift gets the
smooth field value. The `at_limit` flag is unchanged (any presentation at the limit).

## Trust, revised (2026-11-13)

**Analysis** (local re-run of the table chain on 19 fields / 97 presentations / 8580 ROI × presentations, xy-RGCs
2018 + 2022; `MotionDetection` → `MotionRoiMask` (pooled per condition) → `MotionTrust`):
- Within a presentation the old trust varied little (IQR 0.002–0.06) and was barely related to ROI contrast,
  brightness, size, distance to the edge or artifact, vessels, neighbour density or chirp QI (|Spearman ρ| < 0.16).
- `corr_self` contributed nothing: its factor was 1.0 (median) in every presentation, also in GCL1_DN_D of 20220215
  with real z drift. It was ~0.998 even for the dimmest ROIs, i.e. the feared SNR confound did not occur, but neither
  did it detect defocus (the field-level z factor did).
- The per-ROI residual displacement is precise: on the reference recordings median 0.07 px, p90 0.17 px, independent of
  ROI contrast and size.
- In bins with old trust < 0.5 the limiting factor was the per-ROI residual (43 %), the field-level z (21 %), fast
  motion (20 %), coverage (10 %), `corr_self` (6 %).
- The product of soft factors mixed xy and z, and field-level factors (z, fast motion) set every ROI to the same low
  value (20220215 GCL1_DN_D: all ROIs → 0 at the end of the recording although xy is fine).

**Definition** (`compute_trust`, `trust_on_cell`): per ROI and 2 s bin,
`trust` = mean over the frames of the bin of the **fraction of the ROI mask still on its cell**, × the FOV coverage.
The displacement of a frame is the ROI's residual in the bin (local displacement of its band-passed neighbourhood vs.
the field reference, minus the applied shift) plus the frame's deviation of the uncorrected fast motion from the bin
mean. For a binary mask the fraction is the normalized autocorrelation of the mask at that displacement
(|M ∩ (M + d)| / |M|, bilinear between integer lags), so the ROI's size and shape enter: a 3×3 ROI keeps 6/9 at 1 px,
a large ROI more. Separately per bin and field: `dz_um`, the approximate |z change| vs. the recording's mean plane
from the structural-channel similarity (z-stack calibration below; NaN without structural channel). Per repeat:
`trust_rep` (min over the repeat) and `dz_rep` (max). `corr_self` is kept as a diagnostic, not part of trust.
Thresholds (`MotionParams`): `trust_thresh = 0.7` (summaries `frac_low`, `n_rois_flagged`) and `dz_thresh_um = 3`
(`frac_dz_high`); `MotionTrust().valid_repeats(key)` returns the repeats to keep per ROI.

**Behaviour** (same data): per-ROI ranking as before (within-presentation Spearman old vs. new 0.88); new trust
depends on ROI size (ρ 0.35; median 0.966 for ROIs ≤ 13 px vs. 0.980 for > 28 px, 5th percentile 0.84 vs. 0.93).
Noise floor on stable recordings: median ~0.97 (the 0.07 px residual noise), so 1.0 is not reached.

| example | old trust (median ROI) | new trust (median ROI) | dz |
|---|---|---|---|
| 20220201 GCL0 DN_C1, fast motion event at ~185 s (max \|fast\| 0.9 px) | 0.25 | 0.73 (worst ROIs ~0.3) | < 0.6 µm |
| 20220215 GCL1 DN_D, z drift through the plane | → 0 for all ROIs | ~0.93 | up to 4.3 µm, 33 % of bins > 3 µm |
| 20220215 GCL3 DN_D, deformation between conditions | dips to 0.56 at ~170 s | ~0.92 | up to 2.8 µm |

**What it does not capture** (judge these yourself):
- Brightness of what moves in or out: the trace error of a given lost fraction depends on the contrast between the cell
  and its surround relative to the response amplitude. Fast jitter changes the lost fraction from frame to frame and
  creates fluctuations; in the simulation above, 1 px / 1 Hz jitter reduced the trace fidelity to r ≈ 0.38, while the
  mean on-cell fraction of sinusoidal 1 px jitter is 0.76 (13 px ROI) to 0.85 (21 px ROI). Trust above 0.7 does
  therefore not guarantee a clean trace under fast motion; use `fast` (per bin) or a stricter threshold there.
- z per ROI: `dz_um` is field-level (structural channel) and only relative to the recording's mean plane; plane
  changes between presentations are in `MotionDetection` (`z_ref_ch1`) and `MotionTrust.Roi.match_corr`.
- Errors of the reference itself (the mask drawn wrongly, or the reference recording drifting) and motion faster than a
  frame (within-frame line distortions).
- Neuropil contamination and SNR: not motion, not in trust.

## HighRes alignment (`HighResAlignment`, block `highres`, 2026-11-13)

The HighRes scan (`HighRes`) is the same field at 8× the pixel resolution and the same FOV (all local data: 512 vs.
64 px, e.g. 0.184 vs. 1.47 µm/px), usually recorded at the end of the stimulus session; it can be shifted in x, y and
z. `utils.highres_alignment.align_highres` resamples the HR to the field grid (area average; FOVs centred) and
registers it to the channel averages of the ROI mask's presentation (the stored `RoiMask.roi_mask` is aligned to it)
with the motion registration (band-passed Ch0 + Ch1, partially whitened cross-correlation, subpixel). The aligned HR is
resampled once more from full resolution at the shift (no interpolation blur at field resolution). Stored per HR:
`shift_x/y` [field px and µm] (mask position p is at p + shift in the HR), `match_ch0/ch1` (noise-corrected
similarity of the band-passed images after the shift, as the z proxy), `dz_um_approx` (from `match_ch1` with the
z-stack calibration; `match_ch0` without structural channel), `mask_fit_hr` vs. `mask_fit_ref`, the stage differences
and the aligned HR images. The table only reads the database (channel medians), no files.

Validation on all 27 local HR scans (`hr_validate.py`, `hr_fitscan.py`, `zstack_hr.py`; scratch scripts, not in the
repository):

| check | result |
|---|---|
| known shift added to the HR (1.37, −2.21) px | recovered to 0.06 px (median), max 0.22 |
| even vs. odd HR frames | ≤ 0.07 px apart |
| reference = plain channel median (as in the table) vs. registered mean | ≤ 0.11 px apart |
| Ch0-only vs. Ch1-only registration | median 0.35 px apart, max 2.0 (20220201 GCL0 C2, Ch1 match 0.56: plane changed) |
| HR vs. the presentations of the same condition (motion offset) | median 0.5 px (C1) / 0.6 px (C2/D), max 1.2 |
| shift vs. maximum of the ROI-mask fit over a ±3 px grid | median 0.5 px apart (fit surface flat within ±0.5 px); for shifts > 1 px: 0.7 px vs. 1.8 px without alignment |
| stage xy difference vs. measured shift | unrelated (Spearman 0.15): stage positions do not predict the shift |
| HR vs. MB / Chirp of zstack/20190130 (plane offsets 0.56 / 0.71 µm from the z-stack) | match Ch0 0.95, Ch1 0.94 / 0.91 → `dz_um_approx` 1.8 / 2.2 µm |

- Shifts: median 0.7 px, up to 3.4 px (20220201: the stage was moved by ~15 µm between C1 and C2/DETANO; the HRs there
  are shifted by 2.6–3.4 px, and the mask fit goes from negative without alignment to ~0.2 with it).
- Match: C1 HRs Ch0 0.93 (0.85–0.95), Ch1 0.91 (0.76–0.98); HRs of C2/D Ch0 0.87, Ch1 0.75 (0.56–0.97), consistent with
  the plane changes between conditions found above.
- **Precision of `dz_um_approx`**: an HR and a recording of (almost) the same plane do not reach a match of 1 (different
  scan: pixel dwell time, activity, resampling); at 0.56 µm the estimate was 1.8 µm. Read values up to ~2 µm as "same
  plane", and larger values as an upper-biased order of magnitude without sign. A z-stack is needed for real z.
- The precision of the shift for the ROI mask is ~0.5 px (independent criteria agree to that); this is below the
  resolution at which the mask fit can tell.
- `mask_fit_hr` is about half of `mask_fit_ref` (median 0.52) even when aligned: the HR has a different contrast
  (no averaged activity, fewer frames). Compare it between HRs, not with `mask_fit_ref` directly.
- Not applicable to scans with a different FOV or centre (e.g. z-stacks): `align_highres` assumes centred FOVs.

## Open points / next steps

1. ~~Per-ROI shift field between presentations~~ done (see above). Open: within-condition changes of the deformation
   (currently one estimate per condition), and ROIs at the search limit (re-detect? drop?).
2. Calibrate the trust thresholds (`trust_thresh` 0.7, `dz_thresh_um` 3) against trace fidelity (simulated fast
   jitter, real drift), and the `low_contrast` criterion (drug!), on more data / a recording with real within-recording
   drift. The z calibration (similarity → µm) comes from a single z-stack.
3. Fast-motion detection undersamples 1 Hz jitter with 0.5 s bins → try 2-frame bins or per-frame estimates on Ch0+Ch1.
4. ~~Vectorize the per-ROI loops~~ done (trust 20 s → 2.4 s per recording; registration vectorized + cached; fields in parallel).
5. ~~Tables~~ done: `MotionParams`, `MotionReference` (from the presentation of the field ROI mask), `MotionDetection`,
   `MotionRoiMask` (replaces `RoiMaskPresentation` in motion mode; pooling per `cond1` or per presentation),
   `MotionTrust`, and motion-corrected `Traces`. `rf_offset` reads `MotionRoiMask` in motion mode.
6. If presentations are added to a field later, `MotionRoiMask` (and downstream) of that field/condition has to be
   repopulated, because the per-ROI shifts are pooled over the presentations of a condition.

## z-stack check (zstack/20190130/1; `zstack_check.py`, `zstack_time.py`, figures in `out/zstack_20190130/`)

Data: MB (17:46, stage z 4187.5) and Chirp (17:49, z 4186.3) 64×64 @ 1.47 µm/px; HR 512×512 (18:05, z 4184.3, same FOV);
z-stack 512×512×106 (19:44, zoom 0.43 → 0.325 µm/px, 166 µm FOV, start z 4170.3, 1 µm/frame). Setup 2.

Stack z spacing = **1 µm/frame**: frames are not paired; soma z-FWHM 9 frames vs. xy diameter 7.1 µm (→ 9 µm = diameter
+ PSF elongation; 0.5 µm/frame would make somata flatter than wide); and the HR scan lands exactly at its stage z.

| hypothesis | result |
|---|---|
| H1 localisation | ✔ unique xy, z peak at frames 13–15 (Ch0 NCC 0.89–0.92). Caveat: in the stack only ~2 bright cells dominate the match; Ch1 match weak (0.2–0.3). |
| H2 stage refocus = plane change | ✔ for MB→Chirp: plane −1.27 ± 0.05 µm vs. stage −1.2 µm. ✘ for MB→HR: plane −0.56 µm vs. stage −3.2 µm → a ~2.7 µm shift of tissue relative to the stage happened in the break 17:49–18:05 and was compensated by refocusing. **Stage z alone does not tell the imaged plane.** |
| H3 z-proxy calibration | ✔ our Chirp-vs-MB similarity (Ch0 0.96, Ch1 0.94) matches the stack's prediction for 1.27 µm (Ch0 0.91–0.96, Ch1 0.94–0.98). |
| H4 within-recording z | ✔ flat: z(t) sd 0.10–0.11 µm (4 s bins; method precision ~0.05 µm); MB rises +0.46 µm in its last ~10 s. z proxy self ≥ 0.97 agrees; corr(|dz|, proxy) = −0.47 / −0.70 (right sign, tiny range). |
| H5 activity confound | no stimulus-locked z(t) seen, but 4 s bins are too coarse for a strict test. |
| H6 drift until the stack | HR (18:05) at frame 13.9 vs. 14.0 expected from stage → **no z drift 18:05–19:44**. MB/Chirp both 2.7–2.8 µm off their stage-expected frame → the offset arose before the HR scan (same event as in H2). |

**Approximate conversion of the z proxy to µm** (this stack; similarity of band-passed images at recording resolution,
bracket = raw / noise-normalized stack curves; depends on cell size, PSF, objective → treat as order of magnitude):

| similarity | Ch0 | Ch1 (vessels) |
|---|---|---|
| 0.95 | ~1.3–2 µm | ~1–2 µm |
| 0.90 | ~2–2.5 µm | ~2–3 µm |
| 0.75–0.8 | ~3.5–4 µm | ~4 µm |
| 0.5–0.6 | ~6 µm | ~5–6 µm |

Applied (roughly) to the other experiments: C2/D plane changes with Ch1 vs. ref 0.72–0.79 ≈ 4 µm; 20220215 GCL1_MB_D
(0.48) ≈ 6 µm; within-recording drift in GCL1_DN_D (Ch1 self down to 0.75) ≈ 4 µm — i.e. a substantial fraction of a
soma (7–9 µm). C1 non-reference recordings (≥ 0.94–0.99) ≈ ≤ 1.5 µm.
