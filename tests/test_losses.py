import torch

from slassl.losses import SilenceAwareLoss, event_rate_loss, normalized_distillation_loss


def test_silence_loss_handles_completely_silent_sample() -> None:
    loss_fn = SilenceAwareLoss(modes=["near"], min_negatives=4)
    logits = torch.zeros(1, 2, 2, 3, 3, requires_grad=True)
    target = torch.zeros_like(logits)
    loss, stats = loss_fn(logits, target)
    assert torch.isfinite(loss)
    assert stats["negative_bins"] == 4


def test_distillation_stops_teacher_gradient() -> None:
    student = torch.randn(2, 8, requires_grad=True)
    teacher = torch.randn(2, 8, requires_grad=True)
    normalized_distillation_loss(student, teacher).backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_rate_loss_is_zero_for_matching_logits_and_target_rate() -> None:
    logits = torch.zeros(1, 1, 1, 2, 2)
    target = torch.tensor([[[[[0.0, 0.0], [1.0, 1.0]]]]])
    assert torch.isclose(event_rate_loss(logits, target), torch.tensor(0.0))
