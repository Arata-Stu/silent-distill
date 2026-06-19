import pytest

from slassl.cli.finetune import _is_better
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
