# OA-MAE

Data-processing, evaluation and figure-rendering pipeline for the OA-MAE experiments on Sentinel-1 and Sentinel-2 imagery.

This repository holds the experimental harness: area resolution, acquisition discovery, raster standardization, external cloud-support computation, annotation handling, splits, evaluation, deterministic case selection, figure rendering and release validation. The OA-MAE model itself is a separate component, loaded through the adapter interface described below.

The formulation of the method — external cloud support, gated optical-SAR fusion, Cloud-Mix masked pretraining with past-only reconstruction targets, and the change decoder evaluated on the shared support — is given in [`SCIENTIFIC_CONTRACT.md`](SCIENTIFIC_CONTRACT.md).

## Repository structure

```text
.
├── config/                  # project configuration, AOIs, figure contract, model adapters
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

## Models

Methods are declared in [`config/model_adapters.yaml`](config/model_adapters.yaml) and loaded at run time through the adapter interface documented in [`MODEL_ADAPTER_CONTRACT.md`](MODEL_ADAPTER_CONTRACT.md). Each adapter exposes:

```python
def predict_batch(*, batch: dict, checkpoint: str, config: dict) -> dict:
    ...
```

OA-MAE, CROMA and the Hafner baseline are external modules: point `OA_MAE_REPOSITORY`, `CROMA_REPOSITORY` and `HAFNER_REPOSITORY` in `.env` at their checkouts, and the corresponding `*_CHECKPOINT_ROOT` variables at the trained weights. The UNet baselines are internal.

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

`reference_data/` holds the experiment outputs: 600 benchmark samples, leave-one-city-out folds over Dakar, Dar es Salaam, Douala, Garoua, Kigali and Yaounde, few-shot registries for K = 10, 25 and 50, run records, predictions, metrics, annotation audit assets, and the QF1 to QF7 figures. `reference_complements/` adds the deterministic fields derived from those arrays — refined cloud maps, visibility masks, the 16 by 16 support computation, gates, unresolved-positive maps and error maps — documented per record in `reference_complements/manifests/`.

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
