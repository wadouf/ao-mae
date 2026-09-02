from __future__ import annotations

import numpy as np

from oamae_pipeline.observability import compute_support
from oamae_pipeline.metrics import compute


def test_support_identity() -> None:
    a = np.zeros((256, 256), dtype=np.float32)
    b = np.zeros((256, 256), dtype=np.float32)
    _, _, m1, m2, v12 = compute_support(a, b)
    assert np.array_equal(v12, m1 & m2)
    assert v12.all()


def test_metrics_identity() -> None:
    reference = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    probability = np.array([[0.1, 0.9], [0.7, 0.2]], dtype=np.float32)
    support = np.ones_like(reference)
    result = compute(reference, probability, support, 0.5)
    assert result["iou"] == 1.0
    assert result["f1"] == 1.0
