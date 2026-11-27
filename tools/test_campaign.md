# Test campaign ledger

One module per session. Update the row of the module you worked on in the same commit series.
Row format (the coverage script parses `path | status`): `djimaging/<path>.py | done|wip|skip|todo | note`.
`skip` = not worth testing (say why). `wip` = another agent has it, pick another module.

## Already covered by the first campaign (PR 91)

Snippet, filter, trace, preprocess, receptive-field, ScanM/QDSpy reader, filesystem, alias, mask, image,
baden16, storage, exposure-history, response-feature, clustering, classifier and location helpers.
The clustering tables (`djimaging/tables/clustering`) and `RetinalFieldLocationCatTemplate` were removed after
PR 91 together with their tests; don't add rows for them.
Rows below are added as modules are finished.

## Modules

djimaging/utils/snippet_utils.py | done | PR 91
djimaging/utils/mask_utils.py | done | PR 91
djimaging/utils/receptive_fields/temporal_rf_utils.py | done | PR 91
djimaging/utils/receptive_fields/spatial_rf_utils.py | done | PR 91
djimaging/utils/receptive_fields/srf_contour_utils.py | done | PR 91
djimaging/utils/receptive_fields/split_rf_utils.py | done | PR 91
djimaging/tables/core/preprocesstraces.py | done | PR 91
djimaging/tables/core/roi_mask.py | done | before 30%; only module-level pure helpers tested (_roi_mask_source_from_path, _get_autorois_model, _predict_roi_mask, _remove_vessel_rois, load_stack_data, load_default_autorois_models); 0 bugs found; rest lives in table methods needing MySQL (after-coverage not re-measured, .coverage is stale)

## Known suspicions not yet tested

- `Snippets._pad_trace = True` may drop a complete last repetition when the trace ends exactly at its end.
- `srf_dog_model_from_params` is fixed and tested but nothing calls it (candidate for removal).
