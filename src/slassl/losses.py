from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def normalized_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    return (student - teacher).square().sum(dim=-1).mean()


class SilenceAwareLoss(nn.Module):
    def __init__(
        self,
        modes: Sequence[str] = ("random", "near"),
        negative_ratio: float = 3.0,
        negative_weight: float = 1.0,
        near_radius: int = 2,
        min_negatives: int = 256,
    ) -> None:
        super().__init__()
        self.modes = tuple(modes)
        self.negative_ratio = negative_ratio
        self.negative_weight = negative_weight
        self.near_radius = near_radius
        self.min_negatives = min_negatives
        unknown = set(self.modes) - {"random", "near", "hard"}
        if unknown:
            raise ValueError(f"Unknown negative sampling modes: {sorted(unknown)}")

    @staticmethod
    def _shift_without_wrap(
        value: torch.Tensor, amount: int, dimension: int
    ) -> torch.Tensor:
        shifted = torch.zeros_like(value)
        source = [slice(None)] * value.ndim
        destination = [slice(None)] * value.ndim
        if amount > 0:
            source[dimension] = slice(None, -amount)
            destination[dimension] = slice(amount, None)
        else:
            source[dimension] = slice(-amount, None)
            destination[dimension] = slice(None, amount)
        shifted[tuple(destination)] = value[tuple(source)]
        return shifted

    def _candidate_masks(self, positive: torch.Tensor) -> list[torch.Tensor]:
        negative = ~positive
        masks = []
        n, c, b, h, w = positive.shape
        if "random" in self.modes:
            masks.append(negative)
        if "near" in self.modes:
            flat = positive.reshape(n, c * b, h, w).float()
            kernel = 2 * self.near_radius + 1
            near = F.max_pool2d(flat, kernel, stride=1, padding=self.near_radius).bool()
            masks.append(near.reshape_as(positive) & negative)
        if "hard" in self.modes:
            shifted = torch.zeros_like(positive)
            for dimension in (-1, -2, 2):
                shifted |= self._shift_without_wrap(positive, 1, dimension)
                shifted |= self._shift_without_wrap(positive, -1, dimension)
            if c == 2:
                shifted |= positive.flip(1)
            masks.append(shifted & negative)
        return masks

    def _sample_negatives(self, positive: torch.Tensor) -> torch.Tensor:
        candidate_masks = self._candidate_masks(positive)
        selected = torch.zeros_like(positive)
        for batch_index in range(positive.shape[0]):
            positive_count = int(positive[batch_index].sum().item())
            requested = max(self.min_negatives, int(positive_count * self.negative_ratio))
            available = int((~positive[batch_index]).sum().item())
            requested = min(requested, available)
            if requested == 0:
                continue
            quota = (requested + len(candidate_masks) - 1) // max(len(candidate_masks), 1)
            for candidates in candidate_masks:
                unused = candidates[batch_index] & ~selected[batch_index]
                candidate_index = unused.flatten().nonzero().flatten()
                count = min(quota, candidate_index.numel())
                if count == 0:
                    continue
                order = torch.randperm(candidate_index.numel(), device=positive.device)[:count]
                selected[batch_index].view(-1)[candidate_index[order]] = True
            selected_count = int(selected[batch_index].sum().item())
            if selected_count < requested:
                fallback = (~positive[batch_index]) & ~selected[batch_index]
                candidate_index = fallback.flatten().nonzero().flatten()
                count = min(requested - selected_count, candidate_index.numel())
                order = torch.randperm(candidate_index.numel(), device=positive.device)[:count]
                selected[batch_index].view(-1)[candidate_index[order]] = True
        return selected

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        positive = target > 0.5
        selected_negative = self._sample_negatives(positive)
        positive_loss = F.softplus(-logits[positive]).mean() if positive.any() else logits.sum() * 0
        negative_loss = (
            F.softplus(logits[selected_negative]).mean()
            if selected_negative.any()
            else logits.sum() * 0
        )
        loss = positive_loss + self.negative_weight * negative_loss
        stats = {
            "positive_bins": positive.sum().detach(),
            "negative_bins": selected_negative.sum().detach(),
        }
        return loss, stats


def event_rate_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_rate = logits.sigmoid().mean(dim=(1, 2, 3, 4))
    target_rate = target.mean(dim=(1, 2, 3, 4))
    return (predicted_rate - target_rate).abs().mean()


def polarity_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] != 2:
        return logits.sum() * 0
    occupied_count = (target > 0.5).sum(dim=1)
    unambiguous = occupied_count == 1
    if not unambiguous.any():
        return logits.sum() * 0
    class_target = target.argmax(dim=1)
    logits_by_bin = logits.permute(0, 2, 3, 4, 1)
    return F.cross_entropy(logits_by_bin[unambiguous], class_target[unambiguous])
