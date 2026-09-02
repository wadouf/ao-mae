# Model adapter contract

## Adapter entry point

Each method exposes a callable with the logical signature:

```python
def predict_batch(*, batch: dict, checkpoint: str, config: dict) -> dict:
    ...
```

Every method returns `probability` with shape `batch, height, width`, finite float32 values in the interval 0 to 1.

OA-MAE additionally returns, when requested:

```text
cloud_gate
radar_reliability
effective_gate
retrieval_selected_indices
retrieval_selected_product_ids
retrieval_selected_times
retrieval_selected_ages_days
retrieval_target
retrieval_fallback
retrieval_safety_weight
```

## Batch contract

A Stage II batch contains:

```text
sample_id
optical_t1
optical_t2
radar_t1
radar_t2
cloud_t1
cloud_t2
v12
crs
transform
```

Training batches additionally contain `reference`. Stage I batches contain historical optical, historical radar, historical cloud, historical acquisition times, and source product IDs.

## Shapes and types

- Optical: float32, `batch, 10, 256, 256`.
- Radar: float32, `batch, 2, 256, 256`.
- Cloud probability: float32, `batch, 256, 256`, values 0 to 1.
- Support and reference: uint8 or bool, `batch, 256, 256`.
- Probability: float32, `batch, 256, 256`, values 0 to 1.
- Token gates: preserve the native token shape. Store display enlargement metadata separately.

## Checkpoint evidence

Every checkpoint record contains method, fold, K, seed, path, SHA-256, code commit, configuration hash, split hash, normalization hash, best epoch, selection metric, training log path, and environment path.

A filename is not evidence of checkpoint identity.

## Baseline parity

Adapters may convert the shared contract into method-specific inputs. They may not alter the sample crop, reference, common support, held-out-city split, or threshold policy. Every unavoidable adaptation is recorded.

## Diagnostics

Diagnostics are produced during inference and archived at native resolution. A diagnostic may be upsampled only for display. The figure manifest records native shape, display shape, and interpolation.
