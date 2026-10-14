# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# apply_all_penalties for AscendSampler - uses Triton-Ascend kernels.
#
# PATCH-PENALTY-PROMPT-CACHE v2.1 "SLIM+FUSE" (2026-09-30, sjt perf round):
#   On top of v2 (persistent prompt-mask buffer keyed by content hash):
#   ① SLIM: per-step change detection by whole-tensor compare against the
#      previous step's prompt tensor instead of hashing all 160 rows every
#      step. Hash chain (2x ReduceSum over [B,27K] ≈ 0.88ms/step) now runs
#      only for actually-changed rows (usually 0-2). Cost ~0.1ms/step.
#   ② FUSE: output mask is no longer materialized (counts>0 Greater pass
#      ≈0.9ms/step eliminated); apply_all_penalties_kernel computes it
#      in-register from bin_counts (see ops/triton/penalty.py PATCH-SAMPLER-FUSE).
#   Semantics unchanged; VLLM_PENALTY_CACHE_VERIFY=1 keeps exact-token check.

import os

import torch
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad

from vllm_ascend.ops.triton.penalty import _apply_all_penalties_triton
from vllm_ascend.ops.triton.bincount import get_token_bin_counts_and_mask_triton

_CACHE_MAX = 4096
_MASK_CACHE = {}          # prompt-key -> (tokens_compact, mask_row)
_MASK_BUF = None          # [cap, vocab] bool persistent
_LAST_PROMPT = None       # [B, L] int64 copy of last step's prompt tensor
_DUMMY_MASK = None        # [1, vocab] zeros bool placeholder for skip mode
_HASH_W = None
_VERIFY = os.environ.get("VLLM_PENALTY_CACHE_VERIFY", "0") == "1"


def _dummy_mask(vocab_size: int, device: torch.device) -> torch.Tensor:
    """PATCH-SKIP-PROMPT-BINCOUNT: all-False [1, vocab] placeholder used when no
    request has repetition_penalty; prompt side only feeds the repetition branch
    (see apply_all_penalties_kernel), so any value is semantically inert there."""
    global _DUMMY_MASK
    if _DUMMY_MASK is None or _DUMMY_MASK.shape[1] != vocab_size or _DUMMY_MASK.device != device:
        _DUMMY_MASK = torch.zeros((1, vocab_size), dtype=torch.bool, device=device)
    return _DUMMY_MASK


def _hash_weights(L: int, device: torch.device) -> torch.Tensor:
    global _HASH_W
    if _HASH_W is None or _HASH_W.numel() < L or _HASH_W.device != device:
        g = torch.Generator(device="cpu").manual_seed(20260929)
        _HASH_W = torch.randint(-(2**62), 2**62, (max(L, 4096),),
                                generator=g, dtype=torch.int64).to(device)
    return _HASH_W[:L]


def _ensure_buf(B: int, vocab_size: int, device: torch.device) -> None:
    global _MASK_BUF, _LAST_PROMPT
    if _MASK_BUF is None or _MASK_BUF.shape[1] != vocab_size or _MASK_BUF.device != device:
        cap = max(B, 1024)
        _MASK_BUF = torch.zeros((cap, vocab_size), dtype=torch.bool, device=device)
        _LAST_PROMPT = None


def _prompt_mask_cached(prompt_token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Persistent [B, vocab] bool prompt-presence mask; patches changed rows only."""
    global _MASK_BUF, _LAST_PROMPT
    B, L = prompt_token_ids.shape
    device = prompt_token_ids.device
    _ensure_buf(B, vocab_size, device)

    t = prompt_token_ids.to(torch.int64)

    # ① SLIM: whole-tensor compare detects row changes in one ~70MB pass;
    # hash chain below only runs for changed rows (usually 0-2).
    if _LAST_PROMPT is not None and _LAST_PROMPT.shape == t.shape:
        changed = (t != _LAST_PROMPT).any(dim=1).nonzero(as_tuple=True)[0].tolist()
    else:
        changed = list(range(B))
    _LAST_PROMPT = t.clone()

    if not changed:
        return _MASK_BUF[:B]

    sub = t[changed]
    valid = (sub >= 0) & (sub < vocab_size)
    lens = valid.sum(1)
    h = (sub * _hash_weights(L, device)).sum(1)
    keys = [(vocab_size, int(l), int(hh)) for l, hh in zip(lens.tolist(), h.tolist())]

    for j, i in enumerate(changed):
        k = keys[j]
        ent = _MASK_CACHE.get(k)
        if ent is not None:
            toks, m = ent
            if (not _VERIFY) or (toks.numel() == k[1] and bool((toks == t[i, :k[1]]).all())):
                _MASK_BUF[i].copy_(m)
                continue
            _MASK_CACHE.pop(k, None)
        _, m = get_token_bin_counts_and_mask_triton(prompt_token_ids[i:i + 1], vocab_size)
        _MASK_BUF[i].copy_(m[0])
        _MASK_CACHE[k] = (t[i, :int(lens[j])].clone(), m[0].clone())
        while len(_MASK_CACHE) > _CACHE_MAX:
            _MASK_CACHE.pop(next(iter(_MASK_CACHE)))

    return _MASK_BUF[:B]


def _convert_to_tensors(output_token_ids: list[list[int]], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Convert output_token_ids (list of lists) to padded tensor."""
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    skip_prompt_bincount: bool = False,
) -> torch.Tensor:
    """Apply penalties to logits via Triton-Ascend (persistent prompt-mask,
    in-register output mask, no materialized bool pass)."""
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    num_seqs = logits.shape[0]
    # PATCH-SKIP-PROMPT-BINCOUNT: prompt_mask is only consumed by the
    # repetition branch of the kernel; when no request uses repetition
    # penalty, skip the per-step prompt cache bookkeeping (compare pass,
    # hash, sync .tolist()) and pass an inert all-False placeholder.
    if skip_prompt_bincount:
        prompt_mask = _dummy_mask(vocab_size, logits.device).expand(num_seqs, vocab_size)
    else:
        prompt_mask = _prompt_mask_cached(prompt_token_ids, vocab_size)
    output_bin_counts, _ = get_token_bin_counts_and_mask_triton(
        output_tokens_t,
        vocab_size,
        num_seqs,
        need_mask=False,
    )
    _apply_all_penalties_triton(
        logits,
        prompt_mask,
        output_bin_counts,
        repetition_penalties,
        frequency_penalties,
        presence_penalties,
    )
    return logits
