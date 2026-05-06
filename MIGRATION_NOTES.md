# MIGRATION_NOTES.md

Internal record of how `release/` was assembled from the two source repos.
**Remove this file before submission** — it is for the author, not reviewers.

Source repos referenced below:
- **A** = `dir-muvis/` (porting this NeurIPS work; was
  `deconstructing_deep_imbalanced_regression`, locally at
  `/Users/noahpuetz/PythonProjects/dir-muvis`).
- **B** = `MuViS/` (upstream virtual sensing benchmark, locally at
  `/Users/noahpuetz/PythonProjects/MuViS`).

---

## Files copied (source → destination)

### From repo A (`dir-muvis/`)

| Source path                                            | Destination                                            | Notes |
|--------------------------------------------------------|--------------------------------------------------------|-------|
| `dir_methods/losses.py`                                | `src/muvis_dir/methods/losses.py`                      | Verbatim port (header trimmed). |
| `dir_methods/label_distribution.py`                    | `src/muvis_dir/methods/label_distribution.py`          | Verbatim port (header trimmed). |
| `dir_methods/conr.py`                                  | `src/muvis_dir/methods/conr.py`                        | Verbatim port (header trimmed). |
| `dir_methods/rnc.py`                                   | `src/muvis_dir/methods/rnc.py`                         | Verbatim port (header trimmed). |
| `dir_methods/uvote.py`                                 | `src/muvis_dir/methods/uvote.py`                       | Verbatim port; updated relative import. |
| `dir_methods/model_utils.py`                           | `src/muvis_dir/models/backbone.py`                     | Stripped MuViS-config-loading detour: now reads `configs/dataset/<ds>.yaml` directly with a flat `architecture:` block, instead of importing `muvis.utils.architectures.ResNet1D` and resolving `MuViS/configs/<ds>/ResNet1D.yaml`.  Slashes in dataset ids are mapped to `__` for the on-disk filename. |
| `dir_methods/data_utils.py`                            | `src/muvis_dir/data/loader.py`                         | Replaced the `muvis.data_utils.muvis_dataset.MuViSDataset` round-trip with a direct `sktime.datasets.load_from_tsfile` call.  Same 90/10 stratified-random split, same per-channel `StandardScaler`. |
| `dir_methods/trainer.py`                               | `src/muvis_dir/train.py`                               | Restructured into a single CLI-driven driver; logic identical (Adam stepwise schedule, RnC two-stage SGD, UVote gradual α, best-epoch checkpointing on val L1, deterministic seeding).  `CONDA_*` and `MUVIS_*` env-var detours removed.  Config resolution now: `shared.yaml` → `method/<method>.yaml` → `best_per_pair.yaml` → CLI. |
| `evaluation/metrics.py`                                | `src/muvis_dir/eval/metrics.py`                        | Kept only `compute_bins`, `assign_bins`, `bMAE`, `bMASE`.  Dropped `bMSE`, `bR2`, `R2_median`, `MAE`, `MSE`, `per_bin_stats`, `compute_all_metrics` (not needed for the headline table). |
| `configs/dir_methods.yaml`                             | `configs/method/<method>.yaml` × 7                     | Split per-method into one file each. |
| `configs/dir_methods_best_per_pair.yaml`               | `configs/best_per_pair.yaml`                           | Stripped `config_source` / `config` fields (now there is only one method-config flavor); kept `overrides` and merged the v2-vs-v1 distinction into the override fields directly (e.g. v2 LDS becomes `{lds_ks: 9, lds_reweight: inverse, num_bins: 50}`). |
| `results_exp_4.2_multiseed/aggregate.csv`              | `results/expected_bmase.csv`                           | Filtered to the 6 DIR methods (vanilla excluded since `make reproduce` walks 540 = 6 × 9 × 10).  Kept columns `dataset, method, n_seeds, bMAE_mean, bMAE_std`. |

### From repo B (`MuViS/`)

| Source path                                            | Destination                                            | Notes |
|--------------------------------------------------------|--------------------------------------------------------|-------|
| `src/muvis/utils/architectures.py`                     | `src/muvis_dir/models/resnet1d.py`                     | Kept only `ResidualBlock1D` and `ResNet1D`.  Dropped `LSTM`, `MLP`, `LearnablePositionalEncoding`, `Transformer`. |
| `configs/<dataset>/ResNet1D.yaml` × 9                  | `configs/dataset/<dataset>.yaml` × 9                   | Kept only the `Architecture.parameters` block, renamed lower-case `architecture:` for consistency.  REVS subsets get a `__` separator in the filename instead of the slash. |

---

## Files explicitly omitted

### Repo A (`dir-muvis/`)

- `dir_methods/test_backbone.py` — unit-style sanity check, not part of the published pipeline.
- `train_and_save.py` — wrapper around the *baseline* (Experiment 4.1) MuViS model retrain, not used for the DIR results table.
- `evaluation/dataset_stats.py`, `evaluation/aggregate_multiseed.py`, `evaluation/predict.py` — stats / aggregation / re-inference scripts; not needed once predictions are written by the trainer.
- `evaluation/render_paper_outputs.py`, `evaluation/statistical_analysis.py`, `evaluation/tail_density_region.py`, `evaluation/knn_noise_analysis.py` — paper-figure / sensitivity scripts; out of scope.
- `evaluate_exp.42.ipynb`, `evaluate_exp41.ipynb`, `knn_noise.ipynb`, `per_bin_analysis.ipynb`, `sandbox.ipynb`, `tail_density.ipynb`, `target_distribution_figure.ipynb`, `visual_abstract.ipynb` — notebooks, all out of scope per HARD RULE 3.
- `archiv/`, `metric_sensitivity/`, `notebooks/` — exploratory directories.
- `slurm/` — cluster-specific job scripts (would need anonymization of partition names and `${HOME}/PythonProjects` paths).  The Makefile is the portable substitute.
- `tuning/` — the sweep itself (driver scripts + per-candidate prediction files).  The selected configurations are captured in `best_per_pair.yaml`; the sweep machinery is not part of "reproducing the paper table".
- `appendix_reproducibility.tex`, `repro_audit.md`, `repro_assumptions.md` — paper-side documents tracked in the parent repo.
- `figures/`, `*.pdf`, `*.png` — figures and rendered outputs.
- `predictions/`, `results/` (tuning subset), `checkpoints/` — produced artifacts; will be regenerated by `make reproduce`.

### Repo B (`MuViS/`)

- `src/muvis/utils/{datasets.py, logging_utils.py, seeding.py, testing.py, ts.py}` — replaced by inline equivalents in `train.py` (seeding) or by `sktime` directly (`ts.py` is a 1000-line sktime fork; replaced with `sktime.datasets.load_from_tsfile`, which is what the upstream wrapper ultimately delegates to).
- `src/muvis/data_utils/{converters.py, preprocess.py}` — raw-data preprocessing; out of scope (this repo consumes preprocessed `.ts` bundles).
- `src/muvis/train/run_nn_experiments.py`, `src/muvis/train/run_tree_experiments.py` — MuViS baseline trainers (MSELoss, no DIR weighting); out of scope for the DIR results table.
- `configs/<dataset>/{LSTM,MLP,Transformer,Catboost,Xgboost}.yaml` — non-ResNet1D architectures.
- `configs/Tuning/` — upstream HPO sweep configs.
- `data/`, `notebooks/`, `assets/`, `logs/`, `main.py`, `run.sh` — datasets / notebooks / paper figures / driver scripts.

---

## Anonymization changes

- README, citation block, `LICENSE`, and `pyproject.toml` author field replaced with `anonymous`.
- Every Python module header that previously referenced
  `dir_review/agedb_dir_*` paths or "Brandt et al." was rephrased to
  refer to "the AgeDB DIR-review code base" or "upstream MuViS"
  generically.
- Removed: SLURM partition names, `${HOME}/PythonProjects` paths,
  university email, conda env name `dirmuvis`, `wandb`/`mlflow`
  references (none were present in scope, so this was a no-op check
  rather than an active edit).
- Replaced concrete repo URLs in the README with `<repo-url>` /
  `git clone <repo-url>` placeholders.
- `LICENSE` copyright holder is `anonymous`.

---

## Open TODOs

### Data & licensing (must resolve before submission)

- [ ] Dataset hosting URL + SHA-256 manifest for the 18 `.ts` files
  (referenced in README → "Data" as `<TODO>`).
- [ ] Per-dataset license + citation table: port from the MuViS preprint
  Appendix B once it is finalized.
- [ ] Code license: confirm MIT vs Apache-2.0 (currently MIT to match
  MuViS upstream); update `LICENSE` and `pyproject.toml` if changed.
- [ ] Final repo URL in README (the anonymized mirror URL for
  double-blind review).

### Compute reporting

- [ ] Total GPU-hours for `make reproduce` on H100 (referenced in
  README → "Hardware Requirements" and "Reproducing the Paper Table").
  Pull from SLURM accounting once available.

### Results coverage

- [ ] `results/expected_bmase.csv` is missing two rows: `(BeijingPM25Quality,
  conr)` and `(VehicleDynamicsDataset, conr)`.  These pairs are absent
  from the upstream multi-seed aggregate; investigate whether the
  multi-seed run failed for them or whether they should be regenerated
  before the camera-ready.
- [ ] Tolerance choice for `make verify`: the default `--rtol 0.05` is
  generous; a tighter value (e.g. 0.02) is appropriate once
  bitwise-determinism is enabled.  See appendix §A.6 (seed protocol).

### Code

- [ ] `tests/`: empty directory placeholder.  Add at minimum (1) a
  `_assign_bins`/`bMAE` unit test against a fixture and (2) an
  end-to-end smoke test (Vehicle Dynamics, 5 epochs, vanilla) before
  the camera-ready.
- [ ] CLI: ``--lds_kernel`` / ``--lds_reweight`` are exposed but not
  validated against the choices.  Low priority.
- [ ] The "vanilla" baseline is supported by the trainer but is
  excluded from `make reproduce` (which walks the 6 published DIR
  methods, hence the 540 count).  If the camera-ready table includes
  vanilla as a row, change `Makefile :: METHODS` to add it (and the
  `expected_bmase.csv` should grow to 63 rows).
