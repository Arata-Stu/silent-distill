import pytest

from slassl.training import gradient_accumulation_groups


def test_gradient_accumulation_groups_include_partial_final_update() -> None:
    assert gradient_accumulation_groups(10, 4) == [4, 4, 2]


def test_gradient_accumulation_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="accumulation steps positive"):
        gradient_accumulation_groups(10, 0)
