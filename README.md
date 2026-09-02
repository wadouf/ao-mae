# OA-MAE

Reference implementation of OA-MAE, an observability-aligned multimodal self-supervised framework for urban change detection on Sentinel-1 and Sentinel-2 imagery, together with the pipeline used to produce the tables and figures of the paper.

This repository holds the OA-MAE model and the experimental harness around it: area resolution, acquisition discovery, raster standardization, external cloud-support computation, annotation handling, splits, inference, evaluation, deterministic case selection, figure rendering and release validation.

The formulation of the method — external cloud support, gated optical-SAR fusion, Cloud-Mix masked pretraining with past-only reconstruction targets, and the change decoder evaluated on the shared support — is given in [`SCIENTIFIC_CONTRACT.md`](SCIENTIFIC_CONTRACT.md).

## Repository structure

```text
.
├── config/                  # project configuration, AOIs, figure contract, model adapters
├── src/oamae/               # the OA-MAE model: encoders, gates, Cloud-Mix, targets, decoder
├── src/oamae_pipeline/      # library: AOI, discovery, raster, retrieval, metrics, figures
├── scripts/                 # stage scripts 00 to 15, plus the pipeline driver
├── schemas/                 # JSON schemas for every record type
├── templates/               # manifest and annotation table templates
├── tests/                   # contract and figure-policy tests
├── reference_data/          # experiment data: benchmark records, scenes, predictions, figures
├── reference_complements/   # derived deterministic fields, annotations, checkpoint metadata
├── manuscript/              # frozen source tables and the scripts that derive manuscript values
├── outputs/                 # figures, tables, logs, release artifacts
└── package_manifests/       # manifests and SHA-256 checksums
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Conda users can instead create the environment from `environment.yml`.

`.env` holds the credentials for data acquisition (Google Earth Engine or the Copernicus Data Space Ecosystem) and the paths to the model repositories and checkpoint roots. It is never committed. The official endpoints and dataset identifiers are listed in [`SOURCE_REFERENCES.md`](SOURCE_REFERENCES.md).

## The OA-MAE model

The implementation lives in [`src/oamae/`](src/oamae) and follows the formulation in [`SCIENTIFIC_CONTRACT.md`](SCIENTIFIC_CONTRACT.md), equation by equation.

| Module | Contents |
| --- | --- |
| `config.py` | every hyperparameter, marked as fixed by the paper or as an implementation choice |
| `support.py` | token cloud pooling, deterministic refinement, `V12`, the cloud gate and the learned SAR reliability gate (equations 1 to 5) |
| `vit.py` | patch embedding, transformer blocks, and the gated cross-attention block of equation 6 |
| `encoders.py` | the ViT-S/16 optical stream and the ViT-Tiny SAR stream |
| `masking.py` | Cloud-Mix token selection (equation 7) |
| `targets.py` | bounded past-only target construction and the safety weight (equations 8 to 11) |
| `losses.py` | reconstruction, structural fallback, redundancy reduction, focal and Dice (equations 13, 14, 16) |
| `pretrain.py` | Stage I, observability-aligned masked pretraining |
| `segmentation.py` | Stage II, the feature-pyramid adapters, the bi-temporal operator of equation 15 and the residual decoder |
| `inference.py` | the `predict_batch` adapter entry point |

The optical stream uses a ViT-S/16 encoder with 12 blocks, width 384 and 6 heads; the SAR stream uses a ViT-Tiny encoder with 12 blocks, width 192 and 3 heads, linearly projected to the optical width and injected into the final four optical blocks. The Stage-II model holds 30.7 M parameters. The paper reports 29.8 M; the difference comes from submodule widths the paper does not fix — the MLP ratio, the cross-attention width, and the adapter and decoder widths — which are exposed in `OAMAEConfig` rather than hard-coded.

The external support is computed from the cloud product alone. It never depends on model confidence, and `tests/test_model.py` asserts that two differently initialized models return the same `V12`.

### Running a method

Methods are declared in [`config/model_adapters.yaml`](config/model_adapters.yaml) and loaded at run time through the adapter interface documented in [`MODEL_ADAPTER_CONTRACT.md`](MODEL_ADAPTER_CONTRACT.md). Each adapter exposes:

```python
def predict_batch(*, batch: dict, checkpoint: str, config: dict) -> dict:
    ...
```

Point the environment at this checkout and at the trained weights:

```bash
OA_MAE_REPOSITORY=/path/to/ao-mae/src
OA_MAE_CHECKPOINT_ROOT=/path/to/checkpoints
```

Checkpoints are named `oa_mae_loco_{city}_K{k}_seed{NN}.npz` and store `{"config": ..., "state_dict": ...}`. Stage 09 records the SHA-256 of every checkpoint it resolves, and stage 10 records the checkpoint used for each prediction. CROMA and the Hafner baseline stay external through `CROMA_REPOSITORY` and `HAFNER_REPOSITORY`.

An adapter may translate the shared batch contract into method-specific inputs, but not change the sample crop, the reference mask, the common support, the held-out-city split, or the threshold policy.

## Running the pipeline

Configure the cities, paths and policies in `config/project.yaml`, then run the stages in order:

```bash
python scripts/00_preflight.py
python scripts/01_resolve_areas.py          --config config/project.yaml --mode observed
python scripts/02_discover_acquisitions.py  --config config/project.yaml --mode observed
python scripts/03_build_pair_candidates.py  --config config/project.yaml --mode observed
python scripts/04_export_rasters.py         --config config/project.yaml --mode observed
python scripts/05_compute_support.py        --config config/project.yaml --mode observed
python scripts/06_prepare_annotations.py    --config config/project.yaml --mode observed
python scripts/07_validate_annotations.py   --config config/project.yaml --mode observed
python scripts/08_build_splits.py           --config config/project.yaml --mode observed
python scripts/09_train_or_load_models.py   --config config/project.yaml --mode observed
python scripts/10_run_inference.py          --config config/project.yaml --mode observed
python scripts/11_compute_metrics.py        --config config/project.yaml --mode observed
python scripts/12_select_cases.py           --config config/project.yaml --mode observed
python scripts/13_render_figures.py         --config config/project.yaml --mode observed
python scripts/14_validate_release.py       --config config/project.yaml --mode observed
python scripts/15_freeze_release.py         --config config/project.yaml --mode observed
```

`scripts/run_pipeline.py --config config/project.yaml` chains the same stages in a single call.

Stage 04 writes standardized scene bundles under `data/processed/scene_bundles/`; these are regenerated from the archived arrays and are not tracked in git.

## Experiment data

`reference_data/` holds the experiment outputs: 660 tile records over six cities, 480 in the source pool and 180 held out, leave-one-city-out folds over Dakar, Dar es Salaam, Douala, Garoua, Kigali and Yaounde, few-shot registries for K = 10, 25 and 50, run records, predictions, metrics, annotation audit assets, and the QF1 to QF7 figures. `reference_complements/` adds the deterministic fields derived from those arrays — refined cloud maps, visibility masks, the 16 by 16 support computation, gates, unresolved-positive maps and error maps — documented per record in `reference_complements/manifests/`.

## Manuscript values

Every number reported in the paper is derived from the frozen tables under `manuscript/results/source_tables/`, which are byte-identical to the corresponding tables in `reference_data/experiment_benchmark/results/`.

```bash
python manuscript/scripts/build_values.py
```

This regenerates `manuscript/results/values.json`, `values.tex`, `macro_map.json` and `source_map.csv` — 187 values in total. `source_map.csv` maps each manuscript macro to the exact source table it comes from, and `VALUE_CLASSIFICATION.csv` records whether a value is a point estimate, a confidence-interval bound, or a derived quantity.

`manuscript/scripts/verify_results_release.py` checks that a release directory exposes the required figures and value files. `manuscript/checksums/SHA256SUMS.txt` holds the checksums of the archived manuscript package.

## Figures

Figure geometry, panel order, typography and physical dimensions are specified in [`FIGURE_SPECIFICATION.md`](FIGURE_SPECIFICATION.md) and [`FIGURE_DIMENSIONS.csv`](FIGURE_DIMENSIONS.csv), and enforced by `scripts/validate_figure_dimensions.py` and `tests/test_figure_policy.py`.

- English visible text only, ASCII hyphens only
- no watermark, banner or overlay obscuring image pixels or annotation boundaries
- vector PDF and SVG, plus PNG at 300 dpi or higher at the declared physical size
- every panel traceable through a manifest to its source array

## Tests

```bash
pytest tests/
```

## Citation

Wadoufey, A., Bayang Souloukna, P., Namekong Dagha, S., Dayang, P., Kolyang, and Ngakou, A.
*OA-MAE: Observability-Aligned Multimodal Self-Supervised Learning for Reliable Urban Change Detection under Partial Observability.*
EAI Endorsed Transactions.
