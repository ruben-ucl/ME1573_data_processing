# ML Module Dependency Map

Generated: 2026-05-22

---

## Tier 1 — Core Libraries

*Pure modules — no `__main__`, imported by other scripts.*

| Module | Imported by |
|---|---|
| `config.py` | CWT_image_classifier_v3, PD_signal_classifier_v3, final_model_trainer, hyperparameter_tuner, model_tester, delete_test_runs, generate_test_holdout, generate_doe_experiments, comprehensive_hyperopt_analyzer |
| `data_utils.py` | CWT_image_classifier_v3, PD_signal_classifier_v3, final_model_trainer, model_tester |
| `gradcam_utils.py` | final_model_trainer, test_channel_attribution |
| `augmentation.py` | config (lazy), PD_signal_classifier_v3 (lazy) |
| `hyperparameter_registry.py` | parameter_validator |
| `cwt_utils.py` | dataset_labeller, generate_cwt_scalograms |

---

## Tier 2 — Runnable Library

*Has `__main__` AND imported by other scripts.*

| Module | Imported by |
|---|---|
| `training_time_estimator.py` | hyperparameter_tuner (lazy import) |

---

## Tier 3 — Main Pipeline Entry Points

*Have `__main__`, depend on core libs, nothing imports them.*

```
CWT_image_classifier_v3.py   ← config, data_utils
PD_signal_classifier_v3.py   ← config, data_utils, augmentation
final_model_trainer.py        ← config, data_utils, gradcam_utils
hyperparameter_tuner.py       ← config, training_time_estimator
model_tester.py               ← config, data_utils
dataset_labeller.py           ← cwt_utils
generate_cwt_scalograms.py    ← cwt_utils
```

---

## Tier 4 — Utility / Management Entry Points

*Have `__main__`, only depend on `config`.*

```
delete_test_runs.py
generate_test_holdout.py
generate_doe_experiments.py
comprehensive_hyperopt_analyzer.py
```

---

## Tier 5 — Fully Standalone Scripts

*Have `__main__`, no ml/ imports — run independently.*

```
prepare_training_dataset.py
extract_labels_from_hdf5.py
migrate_binary_labels_to_csv.py
migrate_test_results_to_subdirectory.py
convert_test_predictions_to_csv.py
aggregate_test_results.py
visualize_track_predictions.py
regenerate_track_visualizations.py
channel_analysis.py
visualise_model.py
create_dataflow_diagram.py
```

---

## Tier 6 — Possibly Redundant / Orphaned

*Worth reviewing for cleanup.*

| Script | Issue |
|---|---|
| `parameter_validator.py` | Imports `hyperparameter_registry` but nothing imports it and has no `__main__` — currently unreachable |
| `network_architecture_cwt.py` | Architecture diagram generator with a hardcoded path (`D:/ME1573_data_processing/...`), no callers |
| `test_channel_attribution.py` | Imports `gradcam_utils` but no `__main__` block — likely a dev/debug script superseded by `final_model_trainer` |
| `inspect_test_predictions.py` | No `__main__`, no callers — likely a dev-time debug script |
| `__init__.py` | Probably empty; worth confirming |

---

## Key Observations

1. **`config.py` is the single most critical file** — 9 scripts depend on it directly. Changes here have the widest blast radius.
2. **`channel_analysis.py` is designed to be imported** (its own docstring shows `from channel_analysis import analyze_channel_contributions` as the intended usage, and `config.py` has a comment pointing to it), but nothing actually imports it right now — it only runs standalone.
3. **The `old/` subdirectory** contains 10+ archived scripts and is safe to ignore or delete.
4. **`training_time_estimator.py`** is the only file that doubles as both a runnable script and an importable library.
