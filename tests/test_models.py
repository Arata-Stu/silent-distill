import torch

from slassl.models.encoder import TemporalEncoder
from slassl.models.ssl import SLASSLModel


class LossConfig(dict):
    __getattr__ = dict.__getitem__


def test_vit_gru_accepts_event_sequences() -> None:
    config = {
        "vit_patch_size": 4,
        "vit_embed_dim": 64,
        "vit_depth": 4,
        "vit_num_heads": 4,
        "recurrent": "gru",
        "recurrent_hidden_dim": 32,
    }
    encoder = TemporalEncoder("vit_tiny", 4, False, config)
    maps, features = encoder(torch.randn(2, 3, 2, 2, 16, 20))
    assert len(maps) == 4
    assert maps[-1].shape[:3] == (2, 3, 64)
    assert features.shape == (2, 3, 32)


def test_ssl_reports_multi_scale_distillation_losses() -> None:
    model_config = {
        "vit_patch_size": 4,
        "vit_embed_dim": 64,
        "vit_depth": 4,
        "vit_num_heads": 4,
        "multi_scale_distillation": True,
        "distillation_scales": "all",
        "recurrent": "none",
    }
    loss_config = LossConfig(
        lambda_s2l=1.0,
        lambda_silence=1.0,
        lambda_polarity=0.0,
        lambda_rate=0.0,
        negative_modes=["random", "near"],
        negative_ratio=1.0,
        negative_weight=1.0,
        near_radius=1,
        min_negatives=4,
    )
    model = SLASSLModel(
        backbone="vit_tiny",
        input_channels=4,
        polarity_channels=2,
        temporal_bins=2,
        projection_hidden_dim=32,
        projection_dim=16,
        small_stem=False,
        loss_config=loss_config,
        model_config=model_config,
    )
    short = torch.randn(1, 2, 2, 2, 16, 16)
    long = torch.randn_like(short)
    occupancy = torch.zeros(1, 2, 2, 2, 4, 4)
    occupancy[..., 0, 0] = 1
    losses = model(short, long, occupancy)
    assert torch.isfinite(losses["loss"])
    assert {"loss_s2l_block1", "loss_s2l_block4"} <= set(losses)
    expected_diagnostics = {
        "diagnostics/student_std_feature_global",
        "diagnostics/teacher_std_feature_global",
        "diagnostics/student_std_projection_global",
        "diagnostics/teacher_std_projection_global",
        "diagnostics/student_normalized_std_projection_global",
        "diagnostics/teacher_normalized_std_projection_global",
        "diagnostics/student_pairwise_cosine_projection_global",
        "diagnostics/teacher_pairwise_cosine_projection_global",
        "diagnostics/cosine_projection_global",
        "occupancy/predicted_rate",
        "occupancy/target_rate",
        "occupancy/positive_probability",
        "occupancy/negative_probability",
    }
    assert expected_diagnostics <= set(losses)
    assert all(torch.isfinite(losses[key]) for key in expected_diagnostics)
