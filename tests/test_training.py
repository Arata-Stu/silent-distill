import pytest
import torch

from slassl.cli.finetune import _flow_training_loss, _is_better
from slassl.training import gradient_accumulation_groups


def test_gradient_accumulation_groups_include_partial_final_update() -> None:
    assert gradient_accumulation_groups(10, 4) == [4, 4, 2]


def test_gradient_accumulation_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="accumulation steps positive"):
        gradient_accumulation_groups(10, 0)


def test_validation_improvement_respects_direction_and_rejects_nan() -> None:
    assert _is_better(1.0, None, "min")
    assert _is_better(0.9, 1.0, "min")
    assert _is_better(0.9, 0.8, "max")
    assert not _is_better(float("nan"), None, "min")


def test_flow_training_loss_weights_samples_equally() -> None:
    predictions = torch.zeros(2, 2, 1, 2)
    targets = torch.zeros_like(predictions)
    targets[0, 0, 0] = torch.tensor([1.0, 3.0])
    targets[1, 0, 0] = torch.tensor([10.0, 100.0])
    valid = torch.tensor([[[True, True]], [[True, False]]])
    loss = _flow_training_loss(predictions, targets, valid)
    first = (
        torch.tensor(1.0 + 1e-6).sqrt() + torch.tensor(9.0 + 1e-6).sqrt()
    ) / 2
    expected = (first + torch.tensor(100.0 + 1e-6).sqrt()) / 2
    assert float(loss) == pytest.approx(float(expected))
