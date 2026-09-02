# OA-MAE

Reference implementation of **OA-MAE**, an observability-aligned multimodal self-supervised framework for urban change detection under partial observability, together with the full pipeline used to produce the tables and figures of the paper.

The method treats missing optical evidence as a first-class constraint rather than as noise. An external cloud product defines, before any model runs, which pixels are physically observable. Radar is injected into the optical representation only where optical evidence is degraded *and* the radar evidence is structurally reliable. Reconstruction targets are built strictly from earlier acquisitions. Predictions are scored on a shared, method-invariant support, and pixels outside it stay explicitly unresolved instead of being silently mapped to "no change".

- Paper: *OA-MAE: Observability-Aligned Multimodal Self-Supervised Learning for Reliable Urban Change Detection under Partial Observability*, Wadoufey et al., EAI Endorsed Transactions.
- Formal specification of every equation: [`SCIENTIFIC_CONTRACT.md`](SCIENTIFIC_CONTRACT.md).

---

## Contents

1. [Repository structure](#repository-structure)
2. [Installation](#installation)
3. [What the shipped data does and does not cover](#what-the-shipped-data-does-and-does-not-cover)
4. [The data contract](#the-data-contract)
5. [End-to-end reproduction](#end-to-end-reproduction)
6. [The model](#the-model)
7. [Checkpoint identity](#checkpoint-identity)
8. [Operational use](#operational-use)
9. [Experiment data and manuscript values](#experiment-data-and-manuscript-values)
10. [Figures](#figures)
11. [Tests](#tests)
12. [Citation](#citation)

---

## Repository structure

```text
.
├── config/                  # project configuration, AOIs, figure contract, model adapters
├── src/oamae/               # the OA-MAE model: encoders, gates, Cloud-Mix, targets, decoder
├── src/oamae_pipeline/      # pipeline library: AOI, discovery, raster, retrieval, metrics, figures
├── scripts/                 # stage scripts 00 to 15, training 20 to 22, and the driver
├── schemas/                 # JSON schemas for every record type
├── templates/               # manifest and annotation table templates
├── tests/                   # model, contract and figure-policy tests
├── reference_data/          # experiment data: benchmark records, scenes, predictions, figures
├── reference_complements/   # derived deterministic fields, annotations, checkpoint metadata
├── manuscript/              # frozen source tables and the scripts that derive manuscript values
├── outputs/                 # figures, tables, logs, checkpoints, release artifacts
└── package_manifests/       # manifests and SHA-256 checksums
```

## Installation

```bash
git clone https://github.com/wadouf/ao-mae.git
cd ao-mae
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
pytest tests/          # 15 tests, no credentials and no GPU required
```

Conda users can instead create the environment from `environment.yml`. Python 3.11 or later and PyTorch 2.6 or later are required.

`.env` holds the acquisition credentials (Google Earth Engine or the Copernicus Data Space Ecosystem) and the paths to model repositories and checkpoint roots. It is never committed. Official endpoints and dataset identifiers are listed in [`SOURCE_REFERENCES.md`](SOURCE_REFERENCES.md).

```bash
OA_MAE_REPOSITORY=/path/to/ao-mae/src        # this checkout
OA_MAE_CHECKPOINT_ROOT=/path/to/checkpoints  # trained Stage II weights
EARTHENGINE_PROJECT=...
```

## What the shipped data does and does not cover

Read this before planning a reproduction run.

**Included.** The archived experiment outputs: 660 tile records over six cities (480 in the source pool, 180 held out, 110 per city), leave-one-city-out folds, few-shot registries for K = 10, 25 and 50, run records, metrics, prediction arrays, annotation audit assets, the QF1 to QF7 figures, and the frozen tables from which every manuscript value is derived.

**Not included.** Trained weights, and the training tensors themselves. The arrays under `reference_data/` are display and quality-control tensors — 96 by 96, three-band RGB composites, single-channel SAR — whereas the model consumes 256 by 256 tiles with ten Sentinel-2 bands and two Sentinel-1 polarizations. Reproducing the training therefore requires re-acquiring the Sentinel products through stages 01 to 05, which needs Earth Engine or Copernicus credentials.

What you can do without any credentials: install the package, run the tests, inspect and re-derive every manuscript value, read the figure and evaluation contracts, and run the model on your own data once it satisfies the contract below.

## The data contract

Stage scripts exchange `.npz` bundles named `SCN_<city>_<index>.npz`. The city code is the second field of the name and drives the leave-one-city-out logic.

**Stage II bundle** — `data/processed/scene_bundles/`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `optical_t1`, `optical_t2` | `(10, 256, 256)` float32 | Sentinel-2 L2A surface reflectance, bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 |
| `radar_t1`, `radar_t2` | `(2, 256, 256)` float32 | Sentinel-1 GRD VV and VH |
| `cloud_t1`, `cloud_t2` | `(256, 256)` float32 | cloud probability in [0, 1] |
| `reference` | `(256, 256)` uint8 | adjudicated change mask, training and evaluation only |

**Stage I bundle** — `data/processed/pretrain_bundles/`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `optical`, `radar`, `cloud` | as above, single date | the current acquisition |
| `history_optical` | `(T, 10, 256, 256)` float32 | earlier optical acquisitions |
| `history_radar` | `(T, 2, 256, 256)` float32 | earlier radar acquisitions |
| `history_cloud` | `(T, 256, 256)` float32 | cloud probability for each earlier date |
| `history_ages_days` | `(T,)` | age in days of each earlier date, strictly positive |

`src/oamae/data.py` validates these keys and shapes and fails with an explicit message naming the offending file, rather than silently reshaping. Tile size, band count and thresholds are read from `config/project.yaml`, so a different grid only requires editing the configuration.

## End-to-end reproduction

### Step 0 — freeze the environment

```bash
python scripts/00_preflight.py
pip freeze > outputs/logs/environment.txt
git rev-parse HEAD > outputs/logs/commit.txt
```

The commit hash, the configuration hash, the split hash and the normalization hash are all recorded inside every checkpoint. A result that cannot name these four values is not reproducible.

### Step 1 — areas, acquisitions and pairing

```bash
python scripts/01_resolve_areas.py         --config config/project.yaml --mode observed
python scripts/02_discover_acquisitions.py --config config/project.yaml --mode observed
python scripts/03_build_pair_candidates.py --config config/project.yaml --mode observed
```

Tiles are 256 by 256 at 10 m in the local UTM zone. For each optical date the nearest compatible Sentinel-1 acquisition within six days is selected, with the orbit-direction and relative-orbit policy declared in `config/project.yaml`. Product identifiers, sensing times, MGRS tiles, CRS, affine transform, processing baseline and checksums are retained in the sample manifest.

### Step 2 — rasters and the external support

```bash
python scripts/04_export_rasters.py  --config config/project.yaml --mode observed
python scripts/05_compute_support.py --config config/project.yaml --mode observed
```

Stage 05 computes the support of equations 1 to 3: token-level cloud probability by 16 by 16 average pooling, a deterministic refinement that increases conservatism near likely cloud, a per-date visibility mask at the hard threshold 0.85, and their intersection `V12`. This support is external and method-invariant — every compared method is scored on the same pixels.

### Step 3 — labels

```bash
python scripts/06_prepare_annotations.py  --config config/project.yaml --mode observed
python scripts/07_validate_annotations.py --config config/project.yaml --mode observed
```

Open Buildings temporal products propose candidate regions; two annotators independently accept, correct, split, merge or reject them, followed by adjudication. The pipeline prepares and validates these files but does not produce annotation decisions. Because the frame is proposal-driven, a real change never proposed cannot reach verification — hence the independent proposal audit reported alongside the benchmark.

### Step 4 — splits

```bash
python scripts/08_build_splits.py --config config/project.yaml --mode observed
```

Six leave-one-city-out folds. In each fold the target city is excluded from Stage I pretraining and from Stage II supervision, and is used only for testing. Few-shot budgets are K ∈ {10, 25, 50} labeled tile pairs per source city, with ten matched seeds.

### Step 5 — Stage I pretraining

```bash
python scripts/20_pretrain_stage1.py \
    --config config/project.yaml \
    --bundles data/processed/pretrain_bundles \
    --held-out-city dak \
    --epochs 100 --batch-size 8 --learning-rate 1.5e-4 \
    --output outputs/checkpoints/stage1
```

Per-band normalization statistics are computed once over the training pool and written to `outputs/checkpoints/normalization.json`; later runs reuse the file and hash it into every checkpoint. Training combines Cloud-Mix masking at ratio 0.75 over cloud-priority, structurally salient and random tokens, bounded past-only reconstruction targets, the safety weight and the opacity clamp, and the objective of equation 14. Per-epoch metrics go to `outputs/logs/stage1_pretraining.jsonl`, including target availability and fallback rate. `--resume` continues from `stage1_last.pt`.

Run once per fold, so six times. The paper reports 96 GPU-hours of offline pretraining for OA-MAE.

### Step 6 — Stage II few-shot training

```bash
python scripts/21_train_stage2.py \
    --config config/project.yaml \
    --bundles data/processed/scene_bundles \
    --pretrained outputs/checkpoints/stage1/stage1_encoders_dak_seed01.pt \
    --held-out-city dak --few-shot-k 25 --seed 1 \
    --epochs 60 --output outputs/checkpoints/oamae
```

The Stage-I encoders are frozen in the primary setting, which is what isolates representation quality under scarce labels; `--unfreeze-encoders` departs from it and must be reported as such. Supervision is focal plus Dice evaluated only on `V12`, so unobservable pixels never contribute a gradient. The best epoch is selected by validation F1 on source-city tiles held out of the few-shot sample, never on the target city.

The full grid is 6 cities × 3 budgets × 10 seeds = 180 runs:

```bash
for city in dak dar dou gar kig yao; do
  for k in 10 25 50; do
    for seed in $(seq 1 10); do
      python scripts/21_train_stage2.py --config config/project.yaml \
        --pretrained outputs/checkpoints/stage1/stage1_encoders_${city}_seed01.pt \
        --held-out-city $city --few-shot-k $k --seed $seed
    done
  done
done
```

Checkpoints are written as `oa_mae_loco_{city}_K{k}_seed{NN}.npz`, the name stages 09 and 10 expect.

### Step 7 — inference

```bash
export OA_MAE_REPOSITORY=$PWD/src
export OA_MAE_CHECKPOINT_ROOT=$PWD/outputs/checkpoints/oamae
python scripts/09_train_or_load_models.py --config config/project.yaml --mode observed
python scripts/10_run_inference.py --config config/project.yaml --mode observed --few-shot-k 25 --seed 1
```

Stage 09 resolves every method declared in `config/model_adapters.yaml`, verifies that its repository and checkpoint root exist, and records the SHA-256 of each checkpoint. A method that cannot be resolved produces `outputs/observed_run/BLOCKED.json` naming exactly what is missing, and the stage exits non-zero rather than substituting anything. Stage 10 runs inference through the adapter interface and records, for each prediction, the checkpoint used and its checksum.

### Step 8 — metrics

```bash
python scripts/11_compute_metrics.py --config config/project.yaml --mode observed
```

IoU, F1, AUPRC, precision and recall are computed on `V12`, together with coverage, positive coverage and unresolved-positive mass. Every metric is recomputed from the archived arrays; none is transcribed.

### Step 9 — case selection and figures

```bash
python scripts/12_select_cases.py   --config config/project.yaml --mode observed
python scripts/13_render_figures.py --config config/project.yaml --mode observed
python scripts/validate_figure_dimensions.py
```

Case selection is deterministic and frozen before visual inspection, so the qualitative panels cannot be chosen after seeing them. The selection rule retains a median positive gain, an upper-quartile gain, and the largest regression, so failure cases appear by construction.

### Step 10 — compute benchmark

```bash
python scripts/22_benchmark_compute.py --config config/project.yaml \
    --checkpoint outputs/checkpoints/oamae/oa_mae_loco_dak_K25_seed01.npz
```

Reports parameters, operation counts under the convention that one MAC equals two FLOPs, mean and p95 latency, peak VRAM, throughput and model size, with the environment recorded in `outputs/logs/compute_environment.json`. Offline pretraining and online inference are reported separately, and the measurement excludes I/O, preprocessing and cloud masking.

### Step 11 — release

```bash
python scripts/14_validate_release.py --config config/project.yaml --mode observed
python scripts/15_freeze_release.py   --config config/project.yaml --mode observed
```

Validation must report `PASS` before any claim rests on the run. Freezing writes the package manifest and checksums.

`python scripts/run_pipeline.py --config config/project.yaml --mode observed` chains stages 01 to 15; the training scripts 20 to 22 are run separately because they are long and fold-specific.

## The model

The implementation is in [`src/oamae/`](src/oamae) and follows the specification equation by equation.

| Module | Contents |
| --- | --- |
| `config.py` | every hyperparameter, each marked as fixed by the paper or as an implementation choice |
| `support.py` | token cloud pooling, deterministic refinement, `V12`, the cloud gate and the learned SAR reliability gate (equations 1 to 5) |
| `vit.py` | patch embedding, transformer blocks, and the gated cross-attention block of equation 6 |
| `encoders.py` | the ViT-S/16 optical stream and the ViT-Tiny SAR stream |
| `masking.py` | Cloud-Mix token selection (equation 7) |
| `targets.py` | bounded past-only target construction and the safety weight (equations 8 to 11) |
| `losses.py` | reconstruction, structural fallback, redundancy reduction, focal and Dice (equations 13, 14, 16) |
| `pretrain.py` | Stage I |
| `segmentation.py` | Stage II, feature-pyramid adapters, the bi-temporal operator of equation 15, residual decoder |
| `data.py` | bundle datasets, contract validation and frozen normalization |
| `inference.py` | the `predict_batch` adapter entry point |

The optical stream is a ViT-S/16 encoder, 12 blocks, width 384, 6 heads. The SAR stream is a ViT-Tiny encoder, 12 blocks, width 192, 3 heads, linearly projected to the optical width and injected into the final four optical blocks through cross-attention with optical queries and SAR keys and values, scaled token by token by the effective gate.

**Measured against the paper.** The Stage-II model holds 30.712 M parameters and a 122.8 MB state dict, against 29.8 M and 119 MB in Table 6. The operation count measured by `scripts/22_benchmark_compute.py` is 62.5 GFLOPs per bi-temporal pair, against 103.8 in Table 6; that counter covers convolution, linear and attention matmuls and excludes elementwise operations and normalizations. These gaps come from submodule widths the paper does not fix — the MLP ratio, the cross-attention width, and the adapter and decoder widths — which are exposed in `OAMAEConfig` rather than tuned to match the published figures.

**Choices the paper does not fix**, all flagged in `config.py`: MLP ratio 4; the Cloud-Mix split across cloud, structural and random tokens, 50/25/25 by default; the gradient term weight in the reconstruction loss; the formulation of the structural fallback, implemented as gradient-structure agreement with the current radar observation; and the descriptors feeding the SAR reliability gate, implemented as token-mean gradient magnitude and within-token coefficient of variation. Changing any of them changes the model and must be reported.

The external support never depends on model confidence. `tests/test_model.py` asserts that two differently initialized models return the same `V12`, and that the torch implementation agrees exactly with `oamae_pipeline.observability.compute_support`.

### Adding a method

Methods are declared in [`config/model_adapters.yaml`](config/model_adapters.yaml) and loaded at run time through the interface in [`MODEL_ADAPTER_CONTRACT.md`](MODEL_ADAPTER_CONTRACT.md):

```python
def predict_batch(*, batch: dict, checkpoint: str, config: dict) -> dict:
    ...
```

An adapter may translate the shared batch contract into method-specific inputs. It may not change the sample crop, the reference mask, the common support, the held-out-city split, or the threshold policy. CROMA and the Hafner baseline stay external through `CROMA_REPOSITORY` and `HAFNER_REPOSITORY`.

## Checkpoint identity

Every Stage II checkpoint stores `config`, `state_dict` and a `metadata` block carrying the method, fold, budget, seed, selected epoch, selection metric and value, whether the encoders were frozen, the Stage I checkpoint it came from, the code commit, and the configuration, split and normalization hashes, plus the path of its training log.

```python
import torch
payload = torch.load('oa_mae_loco_dak_K25_seed01.npz', weights_only=False)
print(payload['metadata'])
```

A filename is not evidence of checkpoint identity.

## Operational use

The model emits three states, not two:

| State | Condition |
| --- | --- |
| Change | inside `V12`, probability ≥ 0.50 |
| No change | inside `V12`, probability < 0.50 |
| Unresolved | outside `V12` |

```python
state = model.operational_output(out['binary'], out['v12'])
```

Unresolved pixels are not negatives. At the hard threshold 0.85 the paper measures overall coverage 0.620 and positive coverage 0.570, so 43 % of the positive mass falls outside the support: without a follow-up policy, end-to-end recall is 0.376 even though conditional recall on the support is 0.660. A deployment must therefore state what happens to unresolved regions. Human review, SAR-only fallback and deferred reacquisition are evaluated as separate policies with their own workload, accuracy and delay assumptions; they are not inferred from the OA-MAE probability map, and their reported values do not hold unless those quantities are measured in the deployment.

The paper's responsible-use boundary applies to any deployment of this code: the outputs are decision-support layers for urban monitoring, not legal or administrative proof of construction, ownership, demolition or compliance. Fine-grained maps can affect vulnerable communities if interpreted without local validation. Use requires uncertainty-aware communication, human review for consequential decisions, documented label provenance and unresolved regions, and a mechanism for correction and contestation.

## Experiment data and manuscript values

`reference_data/` holds the experiment outputs: 660 tile records over six cities, 480 in the source pool and 180 held out, leave-one-city-out folds over Dakar, Dar es Salaam, Douala, Garoua, Kigali and Yaounde, few-shot registries for K = 10, 25 and 50, run records, predictions, metrics, annotation audit assets, and the QF1 to QF7 figures. `reference_complements/` adds the deterministic fields derived from those arrays — refined cloud maps, visibility masks, the 16 by 16 support computation, gates, unresolved-positive maps and error maps — documented per record in `reference_complements/manifests/`.

Every number reported in the paper is derived from the frozen tables under `manuscript/results/source_tables/`, which are byte-identical to the corresponding tables in `reference_data/experiment_benchmark/results/`:

```bash
python manuscript/scripts/build_values.py
```

This regenerates `values.json`, `values.tex`, `macro_map.json` and `source_map.csv` — 187 values. `source_map.csv` maps each manuscript macro to the exact table it comes from, and `VALUE_CLASSIFICATION.csv` records whether a value is a point estimate, a confidence-interval bound or a derived quantity.

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

Fifteen tests covering the support identity against the pipeline implementation, the refinement bounds, the cloud gate, the Cloud-Mix ratio, past-only eligibility and fallback, the Stage II output contract, the independence of the support from model confidence, the three-state output, support-restricted losses, Stage I differentiability, and the bundle contract validation. They need neither credentials nor a GPU.

## Citation

Wadoufey, A., Bayang Souloukna, P., Namekong Dagha, S., Dayang, P., Kolyang, and Ngakou, A.
*OA-MAE: Observability-Aligned Multimodal Self-Supervised Learning for Reliable Urban Change Detection under Partial Observability.*
EAI Endorsed Transactions.
