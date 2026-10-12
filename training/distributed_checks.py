"""Reject batch branches that would make distributed training collectives diverge."""

import torch
import torch.distributed as dist


def check_batch_participation(data, *, visual_gen, visual_und, device):
    """All ranks must call this before VAE/FSDP forward; no loss values are changed.

    BAGEL computes CE whenever its indexes are not None, including invalid empty
    indexes. MSE reduction requires positive target count; its empty-target model
    branch only supplies a dummy graph. The ViT condition matches BAGEL's forward
    condition because its transformer and connector are separately FSDP-wrapped.
    """
    ce_indexes = data.get("ce_loss_indexes")
    mse_indexes = data.get("mse_loss_indexes")
    ce_count = 0 if ce_indexes is None else ce_indexes.numel()
    mse_count = 0 if mse_indexes is None else mse_indexes.numel()
    flags = torch.tensor([
        ce_count > 0,
        visual_gen and mse_count > 0,
        visual_und and data.get("vit_token_seqlens") is not None
        and data.get("packed_vit_tokens") is not None,
        ce_indexes is not None and ce_count == 0,
    ], dtype=torch.int64, device=device)
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    ce_ranks, mse_ranks, vit_ranks, empty_ce_ranks = flags.tolist()
    advice = "Use the same training variant, observation inputs and image loss masks on every rank."
    if empty_ce_ranks:
        raise ValueError(f"Empty CE loss indexes on {empty_ce_ranks}/{world_size} ranks would "
                         f"produce a zero loss denominator. {advice}")
    participating = {"CE targets": ce_ranks, "MSE targets": mse_ranks, "ViT inputs": vit_ranks}
    mismatched = [f"{name} on {count}/{world_size} ranks"
                  for name, count in participating.items() if count not in (0, world_size)]
    if mismatched:
        raise ValueError("Unsupported heterogeneous batch: " + "; ".join(mismatched) + ". " + advice)
    if ce_ranks == 0 and mse_ranks == 0:
        raise ValueError("Batch has no active CE or MSE targets on any rank. " + advice)
