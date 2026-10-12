"""Eval-matched 50-step GENERATION MSE probe (sokoban hard-OOD), via the PACKED forward.

For each hard "nextframe" probe trajectory (a random imagined frame m, context = the GT envfb history up
to m; the target is the LAST image), run the SAME 50-step flow-matching denoise the eval uses, but on the
packed training forward (FSDP-safe, no KV-cache):

    x_t = noise;  for t,dt in shift-linspace(1,0,T):  v = model(.., gen_velocity=True);  x_t = x_t - v*dt
    eval/mse_gen50       = PIXEL-space mean((vae.decode(x_t) - vae.decode(GT_latent))^2) in [0,1]  (headline)
    eval/mse_gen50_latent = latent-space mean((x_t - GT_latent)^2)                                  (secondary)

Only the LAST image (frame m) is denoised; every other frame (frame0, env-feedback frames, the prior
imagined frames) is forced to CLEAN context (timestep 0) — so the target is generated conditioned on the
TRUE history (teacher-forced), exactly the "1-step-from-truth" world-model accuracy at a random point.
The model's `forward(gen_velocity=True)` returns the velocity at mse_loss_indexes (no re-noise, no sigmoid,
timestep fed directly) — mirroring generate_image/_forward_flow. CFG is off (packed forward = no CFG).

Collective-safe: every rank denoises the SAME number of trajectories x T forwards.
"""
import torch
import torch.distributed as dist


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def _depatchify(v_patch, h, w, p, C):
    """[h*w, p*p*C] -> [C, h*p, w*p] (inverse of the forward's patchify)."""
    x = v_patch.reshape(h, w, p, p, C)
    x = torch.einsum("hwpqc->chpwq", x)        # [C,h,p,w,q]
    return x.reshape(C, h * p, w * p)


@torch.no_grad()
def run_gen50_probe(fsdp_model, ema_model, vae_model, device, training_args, probe_loader,
                    num_timesteps=50, n_per_rank=16, timestep_shift=3.0):
    """Returns {'eval/mse_gen50': float, 'eval/n_gen50': int}. ALL ranks must call this (FSDP collectives)."""
    model_ref = ema_model if getattr(training_args, "use_ema_for_eval", True) else fsdp_model
    bare = _unwrap(model_ref)
    p = int(bare.latent_patch_size)
    C = int(bare.latent_channel)
    # EVAL-MATCHED schedule: the deployed CoT eval (gen_image) uses timestep_shift=3.0, NOT the model's
    # train-time shift (1.0). Use the eval value so the probe MSE tracks the eval generator's footing.
    shift = float(timestep_shift)

    # shift-adjusted schedule (mirror generate_image)
    ts = torch.linspace(1, 0, num_timesteps, device=device)
    ts = shift * ts / (1 + (shift - 1) * ts)
    dts = (ts[:-1] - ts[1:])
    ts = ts[:-1]

    was_training = model_ref.training
    model_ref.train()   # force the packed-sequence training path (like run_eval_mse_ce)
    try:
        # gather this rank's trajectories, padded to EXACTLY n_per_rank (collective alignment)
        rows = []
        it = iter(probe_loader)
        while len(rows) < n_per_rank:
            try:
                rows.append(next(it))
            except StopIteration:
                break
        if not rows:
            return {"eval/mse_gen50": float("nan"), "eval/n_gen50": 0}
        n_real = len(rows)   # only these are averaged; pad rows run for FSDP alignment but are NOT counted
        while len(rows) < n_per_rank:
            rows.append(rows[len(rows) % n_real])   # cycle REAL rows (collective alignment only)

        rank = dist.get_rank() if dist.is_initialized() else 0
        gen = torch.Generator(device=device)
        mse_sum = torch.zeros((), device=device)   # latent-space (secondary)
        pix_sum = torch.zeros((), device=device)   # PIXEL-space reconstruction error (headline)
        cnt = 0
        for idx, batch in enumerate(rows):
            data = batch.cuda(device).to_dict()
            for k in ["batch_data_indexes", "ce_loss_weights", "ce_loss_indexes",
                      "packed_label_ids", "packed_text_labels"]:
                data.pop(k, None)
            if "padded_images" in data and data["padded_images"] is not None:
                data["padded_latent"] = vae_model.encode(data.pop("padded_images"))
            shapes = data["patchified_vae_latent_shapes"]
            h_t, w_t = int(shapes[-1][0]), int(shapes[-1][1])
            n_tok = h_t * w_t
            vae_idx = data["packed_vae_token_indexes"]
            target_pos = vae_idx[-n_tok:]                            # seq positions of the LAST image's tokens
            gt_lat = data["padded_latent"][-1][:, :h_t * p, :w_t * p].detach().clone().float()  # GT target, cropped to this frame

            # gen-mode batch: only the last image is the gen target; everyone else is clean context
            data["mse_loss_indexes"] = target_pos
            ptimes = torch.zeros(vae_idx.shape[0], device=device)

            # deterministic per-trajectory noise -> stable progress curve across eval steps
            gen.manual_seed(0xA5E ^ (rank * n_per_rank + idx))
            x_t = torch.randn(gt_lat.shape, generator=gen, device=device, dtype=torch.float32)
            for i in range(ts.shape[0]):
                data["padded_latent"][-1][:, :h_t * p, :w_t * p] = x_t.to(data["padded_latent"].dtype)
                ptimes[-n_tok:] = ts[i]
                data["packed_timesteps"] = ptimes
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    v_patch = model_ref(**data, gen_velocity=True)   # [n_tok, p*p*C]
                v = _depatchify(v_patch.float(), h_t, w_t, p, C)     # [C,Hl,Wl]
                x_t = x_t - v * dts[i]
            if idx < n_real:                         # count ONLY real trajectories (pads were alignment-only)
                mse_sum += torch.mean((x_t - gt_lat) ** 2)                        # latent-space (secondary)
                # PIXEL-space reconstruction error: decode BOTH latents via the (frozen, local) VAE, MSE in [0,1].
                # vae.decode is NOT FSDP-sharded -> local op, no collective, safe to run on real rows only.
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    vdt = next(vae_model.parameters()).dtype
                    gen_pix = vae_model.decode(x_t.unsqueeze(0).to(vdt))
                    gt_pix = vae_model.decode(gt_lat.unsqueeze(0).to(vdt))
                gen_pix = (gen_pix.float() * 0.5 + 0.5).clamp(0, 1)
                gt_pix = (gt_pix.float() * 0.5 + 0.5).clamp(0, 1)
                pix_sum += torch.mean((gen_pix - gt_pix) ** 2)
                cnt += 1

        cnt_t = torch.tensor(float(cnt), device=device)
        if dist.is_initialized():
            dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(pix_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
        denom = cnt_t.clamp(min=1)
        return {"eval/mse_gen50": (pix_sum / denom).item(),             # PIXEL-space reconstruction error (headline)
                "eval/mse_gen50_latent": (mse_sum / denom).item(),      # latent-space (secondary)
                "eval/n_gen50": int(cnt_t.item())}
    finally:
        if not was_training:
            model_ref.eval()
