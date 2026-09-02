from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from oamae import OAMAEChangeDetector, OAMAEConfig, OAMAEPretrainer
from oamae.masking import cloud_mix
from oamae.support import cloud_gate, observable_support, refined_cloud
from oamae.targets import past_only_target
from oamae.losses import dice_loss, focal_loss
from oamae_pipeline.observability import compute_support


def small_config() -> OAMAEConfig:
    cfg = OAMAEConfig()
    cfg.image_size = 64
    cfg.optical_depth = 4
    cfg.radar_depth = 2
    cfg.fusion_blocks = 2
    cfg.pyramid_layers = (1, 3)
    return cfg


def test_support_matches_pipeline_implementation() -> None:
    rng = np.random.default_rng(0)
    c1 = rng.random((64, 64), dtype=np.float32)
    c2 = rng.random((64, 64), dtype=np.float32)
    _, _, m1, m2, v12 = compute_support(c1, c2)
    result = observable_support(torch.from_numpy(c1).unsqueeze(0), torch.from_numpy(c2).unsqueeze(0))
    assert np.array_equal(result['mask_t1'][0].numpy(), m1.astype(bool))
    assert np.array_equal(result['mask_t2'][0].numpy(), m2.astype(bool))
    assert np.array_equal(result['v12'][0].numpy(), v12.astype(bool))


def test_refinement_is_conservative_and_bounded() -> None:
    cloud = torch.tensor([[[0.0, 0.10], [0.20, 0.90]]]).repeat_interleave(16, -2).repeat_interleave(16, -1)
    tokens, _ = refined_cloud(cloud, 16, 0.20, 0.30)
    assert torch.allclose(tokens[0, 0, 0], torch.tensor(0.0))
    assert torch.allclose(tokens[0, 0, 1], torch.tensor(0.10))
    assert torch.allclose(tokens[0, 1, 0], torch.tensor(0.26))
    assert (tokens <= 1.0).all()


def test_cloud_gate_is_centred_on_the_declared_threshold() -> None:
    value = cloud_gate(torch.tensor([0.50]), alpha=10.0, threshold=0.50)
    assert torch.allclose(value, torch.tensor([0.5]))
    assert cloud_gate(torch.tensor([0.90])) > cloud_gate(torch.tensor([0.10]))


def test_cloud_mix_masks_the_declared_ratio() -> None:
    cfg = small_config()
    tokens = cfg.tokens_per_side() ** 2
    cloud = torch.rand(2, cfg.tokens_per_side(), cfg.tokens_per_side())
    optical = torch.rand(2, cfg.optical_bands, cfg.image_size, cfg.image_size)
    mask = cloud_mix(cloud, optical, cfg.token_size, cfg.mask_ratio, cfg.mask_cloud_fraction, cfg.mask_struct_fraction)
    assert mask.shape == (2, tokens)
    assert (mask.sum(dim=1) == round(cfg.mask_ratio * tokens)).all()


def test_past_only_target_rejects_out_of_window_observations() -> None:
    history_optical = torch.rand(1, 3, 10, 32, 32)
    history_cloud = torch.zeros(1, 3, 32, 32)
    ages = torch.tensor([[10.0, 200.0, -5.0]])
    result = past_only_target(history_optical, history_cloud, ages, token_size=16, maximum_age_days=90)
    selected = result['selected_indices']
    assert (selected[selected >= 0] == 0).all()
    assert not result['fallback'].any()


def test_past_only_target_falls_back_without_eligible_history() -> None:
    history_optical = torch.rand(1, 2, 10, 32, 32)
    history_cloud = torch.zeros(1, 2, 32, 32)
    ages = torch.tensor([[500.0, 900.0]])
    result = past_only_target(history_optical, history_cloud, ages, token_size=16, maximum_age_days=90)
    assert result['fallback'].all()


def test_change_detector_output_contract() -> None:
    cfg = small_config()
    model = OAMAEChangeDetector(cfg).eval()
    size = cfg.image_size
    batch = dict(
        optical_t1=torch.rand(1, 10, size, size), optical_t2=torch.rand(1, 10, size, size),
        radar_t1=torch.randn(1, 2, size, size), radar_t2=torch.randn(1, 2, size, size),
        cloud_t1=torch.rand(1, size, size), cloud_t2=torch.rand(1, size, size),
    )
    with torch.no_grad():
        out = model(**batch)
    probability = out['probability']
    assert probability.shape == (1, size, size)
    assert torch.isfinite(probability).all()
    assert probability.min() >= 0 and probability.max() <= 1
    assert torch.equal(out['binary'], probability >= cfg.probability_threshold)


def test_support_does_not_depend_on_model_confidence() -> None:
    cfg = small_config()
    size = cfg.image_size
    cloud_t1, cloud_t2 = torch.rand(1, size, size), torch.rand(1, size, size)
    batch = dict(
        optical_t1=torch.rand(1, 10, size, size), optical_t2=torch.rand(1, 10, size, size),
        radar_t1=torch.randn(1, 2, size, size), radar_t2=torch.randn(1, 2, size, size),
        cloud_t1=cloud_t1, cloud_t2=cloud_t2,
    )
    external = observable_support(cloud_t1, cloud_t2, cfg.token_size, cfg.seed_threshold, cfg.refinement_delta, cfg.hard_threshold)
    for seed in (0, 1):
        torch.manual_seed(seed)
        with torch.no_grad():
            out = OAMAEChangeDetector(cfg).eval()(**batch)
        assert torch.equal(out['v12'], external['v12'])


def test_operational_output_marks_unresolved_pixels() -> None:
    binary = torch.tensor([[[True, False], [True, False]]])
    v12 = torch.tensor([[[True, True], [False, False]]])
    state = OAMAEChangeDetector.operational_output(binary, v12)
    assert state.tolist() == [[[1, 0], [2, 2]]]


def test_supervised_losses_ignore_pixels_outside_the_support() -> None:
    probability = torch.tensor([[[0.9, 0.9], [0.9, 0.9]]])
    reference = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    inside = torch.tensor([[[True, True], [False, False]]])
    everywhere = torch.ones_like(inside)
    assert focal_loss(probability, reference, inside) < focal_loss(probability, reference, everywhere)
    assert dice_loss(probability, reference, inside) < dice_loss(probability, reference, everywhere)


def test_pretrainer_objective_is_finite_and_differentiable() -> None:
    cfg = small_config()
    cfg.decoder_depth = 1
    model = OAMAEPretrainer(cfg)
    size = cfg.image_size
    batch = dict(
        optical=torch.rand(1, 10, size, size), radar=torch.randn(1, 2, size, size), cloud=torch.rand(1, size, size),
        history_optical=torch.rand(1, 2, 10, size, size), history_radar=torch.randn(1, 2, 2, size, size),
        history_cloud=torch.rand(1, 2, size, size) * 0.1, ages_days=torch.tensor([[10.0, 40.0]]),
    )
    result = model.loss(batch)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_dataset_rejects_a_bundle_that_breaks_the_contract(tmp_path) -> None:
    from oamae.data import BundleContractError, ChangeDataset

    path = tmp_path / 'SCN_dak_0001.npz'
    np.savez_compressed(path, s2_rgb_t1=np.zeros((96, 96, 3), dtype=np.float32))
    dataset = ChangeDataset([path], small_config())
    with pytest.raises(BundleContractError, match='missing'):
        dataset[0]


def test_dataset_reads_a_conforming_bundle(tmp_path) -> None:
    from oamae.data import ChangeDataset

    cfg = small_config()
    size = cfg.image_size
    path = tmp_path / 'SCN_dak_0001.npz'
    np.savez_compressed(
        path,
        optical_t1=np.zeros((10, size, size), dtype=np.float32),
        optical_t2=np.zeros((10, size, size), dtype=np.float32),
        radar_t1=np.zeros((2, size, size), dtype=np.float32),
        radar_t2=np.zeros((2, size, size), dtype=np.float32),
        cloud_t1=np.zeros((size, size), dtype=np.float32),
        cloud_t2=np.zeros((size, size), dtype=np.float32),
        reference=np.zeros((size, size), dtype=np.uint8),
    )
    item = ChangeDataset([path], cfg)[0]
    assert item['sample_id'] == 'SCN_dak_0001'
    assert item['optical_t1'].shape == (10, size, size)
    assert item['reference'].shape == (size, size)
