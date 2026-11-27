# RF vs ROI position: slopes below 1 (open investigation)

Status: open, started 2026-11-19.

## Observation

`RfRoiOffset().plot()` plots the RF center position in field coordinates (`roi + offset`) against the ROI position.
If the geometry is correct and the RF centers are unbiased, the slope should be 1. It isn't.

Panel 5 is the affine diagnostic, `fit_affine_roi_to_rf` in
[rf_offset.py](../djimaging/tables/receptivefield/rf_offset.py). It fits `rf ≈ A·roi + b_field`, with one intercept per
field, and splits `A` into rotation, per-axis scale and shear.

| Data | rot | scale x | scale y | shear | R² | n |
|---|---|---|---|---|---|---|
| `split_qidx > 0.55`, 1 experiment | (no affine yet; regression slopes) | ~0.75 | ~0.55 | | | ~120 |
| lower quality threshold, 8 fields | +7.8° | 0.50 | 0.41 | −0.01 | 0.06 | 279 |
| higher quality threshold, 8 fields | +5.6° | 0.76 | 0.53 | −0.03 | 0.68 | 92 |

The middle row is not valid. Failed or edge-clipped RF fits at ±200–300 µm dominate it, and the fit isn't robust to
them. Apply a quality threshold first instead of trimming outliers in the fit.

What we know from the valid rows:
- **The rotation is too small to matter.** cos 5.6° = 0.995, so rotation can't explain the slopes.
- **Shear is about 0.**
- **The scales are below 1 and differ between axes** (x about 0.76, y about 0.53, ratio about 0.7). This holds across
  datasets and across most fields; one per-field fit is degenerate, probably because that field has few ROIs.
- **The mean offset is about 0**, so the centering and offsets look fine.

Working hypothesis: one of the µm scales is wrong per axis. Either the ROI positions are spread too wide, or the RF
positions too narrow. Shrinkage in the RF estimate would normally affect x and y equally, so it's less likely.

## Plan

### 1. Imaging pixel size (ROI side)

`RelativeRoiLocationWrtField.make` ([roi_location.py:86](../djimaging/tables/location/roi_location.py#L86)) uses a single
`Field.pixel_size_um` for both axes. That value is `scan_width_um / nxpix / zoom` (see `setup_utils.get_pixel_size_xy_um`,
`exp_history_utils`), so square pixels are assumed. Check:
- [ ] Is the y extent (line spacing, galvo amplitude) really equal to the x pixel size × nypix? Check ScanM headers
      for different nx/ny and for y amplitude settings, and for fields with nx ≠ ny.
- [ ] The x/y swap and signs: `relx = -d2_um`, `rely = -d1_um` (the mask is "already rotated to normal view").
      Check that d1 and d2 aren't swapped relative to which axis has which pixel size.
- [ ] `get_rel_roi_pos` in [roi_utils.py](../djimaging/utils/scanm/roi_utils.py) with `ang_deg`: how is the rotation
      applied, and is it applied before or after scaling?
- [ ] Is `scan_width_um` for the setup or setups in this dataset correct? Compare with a calibration grid or bead image
      if one exists.

### 2. Stimulus pixel size (RF side)

`RfOffsetTemplate.fetch_and_compute` ([rf_offset.py](../djimaging/tables/receptivefield/rf_offset.py)) converts the
Gaussian center to µm with `stim_dict['pix_scale_x_um']` / `['pix_scale_y_um']`. `compute_rf_roi_offset` then maps it
into the field frame with `array_to_field_matrix(array_to_screen, screen_x_axis, screen_y_axis)`.
- [ ] Do `pix_scale_x_um` and `pix_scale_y_um` match the real checker size on the retina for this stimulator?
- [ ] Does `array_to_screen` or the stimulator geometry swap the axes, so that the x scale is applied to field y or the
      other way round? A swap combined with non-square checkers would produce exactly this kind of per-axis mismatch.
- [ ] Check `srf.shape` against the stimulus array shape: is `x_mean` paired with `shape[1]`, and `y_mean` with
      `shape[0]`?

### 3. Separate a fixed factor from noise

- [ ] Fit the affine map per setup, per zoom and per stimulus. If the scales stay the same (about 0.76 / 0.53), the cause
      is a fixed calibration factor; work out which constant would give 1.
- [ ] If the scales change with zoom, look at the imaging side (step 1). If they change with stimulus or stimulator,
      look at the stimulus side (step 2).
- [ ] Fit with a scale forced to be equal in x and y, and compare the residuals with the free fit, to confirm the
      difference between axes is real.

### 4. Rule out RF estimation shrinkage

Only if steps 1–3 don't explain it.
- [ ] Use the RF contour center (`RfContours`) instead of the Gaussian fit, and see whether the slopes change.
- [ ] Use a simulated ROI with a known RF position through the same STA and fit pipeline.

## Notes

- To reproduce:
  `plot_if_data(RfRoiOffset() & (SplitRF() & "split_qidx>0.55").keys(), method='plot')`, or use a higher quality
  threshold.
- Always apply a quality threshold before reading the affine numbers, because RFs at the stimulus border wreck the
  least-squares fit.
- Related recent changes: the `RetinaOrientation` table (commit 934aff4) and the stimulator transforms in
  `djimaging/utils/stimulator.py`.
