"""Continuation-logprob readout for activation-patching runners (torch).

The patching steps score a *shared* continuation span under (possibly batched) logits
from a patched forward pass. This is the torch counterpart of the pure-list metrics in
``axiom.metrics.continuation`` and is shared by every patching runner so the readout is
defined once.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def continuation_logp(logits: "torch.Tensor", target_ids: "torch.Tensor", cont_start: int) -> "torch.Tensor":
    """Sum of continuation-token log-probabilities per batch row.

    Args:
        logits: ``[B, T, V]`` logits from a (patched) forward pass over the target sequence.
        target_ids: ``[T]`` the shared sequence being scored (the target/reference ids).
        cont_start: index in ``target_ids`` where the continuation (readout) begins.

    Returns:
        ``[B]`` summed ``log P(continuation tokens)`` for each row. Every row is scored
        against the same ``target_ids`` (the patch changes activations, not tokens).
    """
    import torch

    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = target_ids[1:]  # next-token targets
    tok_lp = lp[:, :-1, :].gather(-1, tgt.view(1, -1, 1).expand(lp.size(0), -1, 1)).squeeze(-1)  # [B, T-1]
    seg = tok_lp[:, cont_start - 1 : target_ids.shape[0] - 1]  # continuation token logprobs
    return seg.sum(dim=1)
