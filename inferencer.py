# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Modified for Agima: documentation-only removal of machine-specific references.

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch
import os
from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

COT_GEN_SYSTEM_PROMPT = '''You should generate the image step by step. 
For example, if you want to generate a Mario without a red hat, you can first generate a Mario, and then remove the red hat. 
When you think you have already generated the image, you can generate "STOP".
The process is enclosed within <think> </think> tags, 
i.e. <think> generate mario first </think> image of mario here <think> remove his hat </think> image of mario without hat here <think> STOP </think>'''

REFLECTION_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 

Now, look at the original instruction, the actions you have taken, and the final resulting generated image. 
What is the next action you should take to generate the image that matches the instruction? If the last image already satisfies the instruction, you can generate "STOP.".
The final answer should be a concise and single sentence instruction.

The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here.'''

REFLECTION_SYSTEM_PROMPT_ACTION= '''What is the next action you should take to generate the image that matches the instruction? 
Reply a single sentence instruction.'''

REFLECTION_SYSTEM_PROMPT= '''
First, describe the current image.
Then, check if the current image matches the instruction.
What is the next action you should take to generate the image that matches the instruction?
If the current image already matches the instruction, you can generate "STOP.".
Reply a single sentence instruction.'''

REFLECTION_SYSTEM_PROMPT_MULTI_TURN_SIMPLE= '''
You will be given a sequence of images and text instructions. 
This will contain the orgininal image, the actions you have taken and the resulting image at each step.
Is the final images correct? If not, what is the next action you should take to generate the image that matches the instruction?
If the final image is correct, you can generate "STOP.".
You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here
'''

REFLECTION_SYSTEM_PROMPT_MULTI_TURN = '''
You are given two aligned sequences:
- Images: I_0, I_1, ..., I_T  (I_0 is the original; I_k is the result after action A_k)
- Actions: A_1, A_2, ..., A_T  (text actions applied in order)
Optionally, an instruction may also be provided.

Your goal:
1) Verify whether I_T is the correct outcome of applying A_1..A_T to I_0. If an instruction is present, I_T must also satisfy it.
2) If I_T is incorrect, propose the single next action to apply to I_T so it becomes the correct outcome implied by A_1..A_T (and any instruction, if given).
3) If I_T is correct, generate the action as "STOP".

Constraints for the final answer:
- If a fix is needed, output exactly one action (no extra text).
- Use the same action vocabulary/units as in A_1..A_T.
- Prefer the minimal fix (e.g., undo/adjust a magnitude, or select a prior image such as “add a mustache to the third image”).
- Be brief with your reasoning.

The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here.
'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers), # KV Cache
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True, omit_trailing_eoi=False):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
                omit_trailing_eoi=omit_trailing_eoi,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )
        # unpacked_latent[0] = unpacked_latent[0].to(torch.bfloat16)
        image = self.decode_image(unpacked_latent[0].to(torch.bfloat16), image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        latent = latent.to(self.model.device)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0, end_token_id=None):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if end_token_id is None:
            end_token_id = self.new_token_ids['eos_token_id']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=end_token_id,
            **generation_input,
        )
        # Decode the generated tokens
        # prepare_start_tokens adds BOS, and generate_text stops at EOS
        # So unpacked_latent contains [BOS, ..., generated_text, EOS]
        # We need to decode and remove BOS/EOS tokens
        output_token_ids = unpacked_latent[:,0].tolist()

        # Remove BOS token if present (it's added by prepare_start_tokens)
        if output_token_ids and output_token_ids[0] == self.new_token_ids['bos_token_id']:
            output_token_ids = output_token_ids[1:]

        # Remove EOS token if present (added by generate_text when stopping)
        if output_token_ids and output_token_ids[-1] == self.new_token_ids['eos_token_id']:
            output_token_ids = output_token_ids[:-1]

        # Decode the cleaned token ids
        if output_token_ids:
            output = self.tokenizer.decode(output_token_ids)
        else:
            output = ""
        return output

    @torch.no_grad()
    def gen_cot_chunk_span(self, gen_context, *, image_shape=(256, 256), max_tokens=600,
                           cfg_text_scale=1.0, cfg_img_scale=1.0, num_timesteps=50,
                           timestep_shift=3.0, cfg_interval=(0.4, 1.0),
                           do_sample=False, temperature=1.0, frame_callback=None,
                           frame_provider=None):
        """Generate ONE assistant CoT chunk as a SINGLE continuous <|im_start|>..<|im_end|> span (v4).

        Native token-by-token via model.generate_step: emit text; when the model predicts <|vision_start|>,
        denoise an imagined frame (gen_image) and feed it back into the SAME KV cache via
        update_context_image(vae+vit), then CONTINUE in the same span until <|im_end|>. Mutates gen_context
        IN PLACE so the caller carries the cache across chunks. Returns (chunk_text, [PIL frames]).

        This is the v4 single-span analogue of the old gen_text(stop=im_end)+gen_image per-segment loop
        (which only worked for per-segment ckpts that emit an im_end after each frame; on v4 it never sees
        the intermediate <|vision_start|> and generates ~0 frames).

        Marker/rope handling (empirically grounded against the v4 training token dump). Per frame training is
            .. text <|vision_start|>(text)  <|vision_start|>[vae]<|vision_end|> <|vision_start|>[vae]<|vision_end|> <|vision_start|>[vit]<|vision_end|> ..
        We (a) COMMIT the predicted (text) <|vision_start|>; (b) gen_image; (c) update_context_image(vae,vit)
        commits one vae block + one vit block (skipping the attention-masked vae loss-target, as the existing
        allstep inference does), with the vit block's TRAILING end_of_image OMITTED and its rope slot held;
        (d) feed end_of_image as the next generate_step query -> the post-frame text is predicted with no
        doubled marker and no rope drift, reproducing the trained [.. vit, end_of_image, next-text]. CFG OFF.
        """
        SOI = self.new_token_ids['start_of_image']     # <|vision_start|>
        EOI = self.new_token_ids['end_of_image']       # <|vision_end|>
        IM_END = self.new_token_ids['eos_token_id']    # <|im_end|>
        BOS = self.new_token_ids['bos_token_id']       # <|im_start|>
        dev = self.model.device
        cfg_ph = self.init_gen_context()               # placeholder cfg precontext, unused at scale 1.0

        def _step(prev):
            flag, np_, nk, nr, nt = self.model.generate_step(
                gen_context['past_key_values'], gen_context['kv_lens'], gen_context['ropes'],
                self.new_token_ids, prev_token=prev, do_sample=do_sample, temperature=temperature)
            gen_context['past_key_values'] = np_
            gen_context['kv_lens'] = nk
            gen_context['ropes'] = nr
            return flag, nt

        out_token_ids = []
        imgs = []
        prev_token = None                              # first step feeds BOS -> opens the assistant <|im_start|> turn
        _TOKDBG = os.environ.get("BAGEL_COT_TOKEN_DEBUG", "0") == "1"   # per-token trace (verification)
        _DBG_CAP = 120
        n_dbg = 0
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            for _ in range(max_tokens):
                flag, next_tokens = _step(prev_token)
                if flag == 'vision_start':
                    if _TOKDBG and n_dbg < _DBG_CAP:
                        print(f"[TOK#{n_dbg:03d}] MODEL-PRED= <|vision_start|>  -> the MODEL itself requests an image; will gen frame #{len(imgs)}", flush=True); n_dbg += 1
                    out_token_ids.append(SOI)
                    _step(next_tokens)                 # commit the predicted (text) vision_start; drop its prediction
                    if frame_provider is not None:
                        prefix = self.tokenizer.decode([t for t in out_token_ids if t != BOS])
                        img = frame_provider(prefix, len(imgs))
                    else:
                        img = self.gen_image(
                            image_shape, gen_context,
                            cfg_text_precontext=cfg_ph, cfg_img_precontext=cfg_ph,
                            cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval, num_timesteps=num_timesteps, timestep_shift=timestep_shift)
                        if img.size != tuple(image_shape):
                            img = img.resize(image_shape, Image.Resampling.LANCZOS)
                        if frame_callback is not None:
                            prefix = self.tokenizer.decode([t for t in out_token_ids if t != BOS])
                            img = frame_callback(img, prefix, len(imgs))
                    if img.size != tuple(image_shape):
                        img = img.resize(image_shape, Image.Resampling.LANCZOS)
                    imgs.append(img)
                    ri = pil_img2rgb(img)
                    # Trained frame = vae-target(loss=1,masked) + vae-ctx(loss=0) + vit(loss=0) (3 blocks).
                    # TRAINING rope (dataset_base.py:529-533), p = rope of the predicted <|vision_start|>;
                    # the text run leaves curr_rope_id = p+1 entering the frame:
                    #   vae-target: tokens @ p+1, loss==1 -> NO advance (target shares its slot with vae-ctx)
                    #   vae-ctx:    tokens @ p+1, loss==0 -> advance +1 -> p+2
                    #   vit:        tokens @ p+2, advance +1 -> p+3
                    #   end_of_image (next query) @ p+2, next text @ p+3
                    # i.e. the VISIBLE frame sits at vae-ctx=p+1, vit=p+2; eoi queries at p+2, next-text at p+3.
                    #
                    # Per-frame commit modes (entering with gen_context['ropes']==[p+1] after committing the
                    # predicted vision_start; CPU-verified in test_cot_exact_rope.py):
                    #   exact:    NO rope skip. prepare_vae_images puts vae-ctx @ p+1 (then +1 -> p+2);
                    #             prepare_vit_images(omit_trailing_eoi) puts vit @ p+2 and does NOT advance
                    #             (rope held at p+2); end_of_image fed as next query @ p+2 -> next text @ p+3.
                    #             Reproduces the trained visible-frame layout EXACTLY (0 drift). The correct path.
                    #   ropeskip: adds +1 BEFORE committing (a slot for the masked target), so vae-ctx lands @ p+2,
                    #             vit @ p+3, eoi @ p+3, next text @ p+4 -> +1 drift on the self-fed frame EVERY
                    #             frame (drift compounds across frames). Kept only for comparison.
                    #   3block:   also commits the vae loss-target as a visible block -> same +1 drift as ropeskip,
                    #             plus the masked target becomes attendable. Kept only for comparison.
                    # NOTE: the previous comment here claimed "ropeskip matches training / 2block drifts -1" --
                    # that is BACKWARDS (see GT trace above and test_cot_exact_rope.py). 'exact' is rope-correct.
                    # The last committed block's trailing end_of_image is omitted + fed as the next query.
                    _mode = os.environ.get("BAGEL_COT_FRAME_MODE", "exact")
                    if _mode == "3block":
                        self.update_context_image(ri, gen_context, vae=True, vit=False)            # vae loss-target
                    elif _mode == "ropeskip":
                        gen_context['ropes'] = [gen_context['ropes'][0] + 1]                       # skip masked target's rope slot
                    # exact / 2block: no rope skip -> vae-ctx lands at p+1 exactly like training.
                    self.update_context_image(ri, gen_context, vae=True, vit=True, omit_trailing_eoi=True)  # vae-ctx + vit
                    prev_token = torch.tensor([EOI], dtype=torch.long, device=dev)
                    if _TOKDBG and n_dbg < _DBG_CAP:
                        print(f"[TOK#{n_dbg:03d}]   <-- fed SELF-GENERATED frame #{len(imgs)-1} back into context; next query = <|vision_end|>", flush=True); n_dbg += 1
                    continue
                tok = int(next_tokens[0].item())
                if _TOKDBG and n_dbg < _DBG_CAP:
                    dec = self.tokenizer.decode([tok])
                    mark = "   <|im_end|> -> STOP (turn ends)" if tok == IM_END else ""
                    print(f"[TOK#{n_dbg:03d}] MODEL-PRED= {dec!r}  (id={tok}){mark}", flush=True); n_dbg += 1
                if tok == IM_END:
                    break                              # assistant turn closed
                out_token_ids.append(tok)
                prev_token = next_tokens
        text = self.tokenizer.decode([t for t in out_token_ids if t != BOS]) if out_token_ids else ""
        return text, imgs

    @torch.no_grad()
    def gen_text_with_logprobs(self, gen_context, max_length: int = 500, do_sample: bool = True,
                                temperature: float = 1.0, end_token_id=None):
        """Same as `gen_text` but also returns the per-token log-probs of the
        emitted sequence under the policy at sampling time. Used by GRPO/RL
        rollouts to compute π_θ_old(action_tokens | state).

        Returns:
            output_text: decoded text (BOS/EOS stripped)
            output_token_ids: list[int] of the *kept* tokens (BOS/EOS stripped),
                matches output_text 1:1
            output_logprobs: torch.Tensor of shape (len(output_token_ids),) on CPU,
                log p(token_i | <prefix, tokens_0..i-1>) at T=1.0
        """
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if end_token_id is None:
            end_token_id = self.new_token_ids['eos_token_id']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        tokens, logprobs = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=end_token_id,
            return_logprobs=True,
            **generation_input,
        )
        # tokens / logprobs are aligned step-by-step. Strip BOS (added by
        # prepare_start_tokens) and EOS (sampled stop token, if any).
        token_ids = tokens[:, 0].tolist()
        lp = logprobs[:, 0].detach().cpu()

        bos = self.new_token_ids['bos_token_id']
        eos = self.new_token_ids['eos_token_id']

        start = 1 if (token_ids and token_ids[0] == bos) else 0
        end = len(token_ids)
        if token_ids and token_ids[-1] == eos:
            end -= 1

        kept_ids = token_ids[start:end]
        kept_lp = lp[start:end]

        text = self.tokenizer.decode(kept_ids) if kept_ids else ""
        return text, kept_ids, kept_lp
        
    @torch.no_grad()
    def interleave_inference_multi_turn(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        max_steps=4,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

            system_prompt = COT_GEN_SYSTEM_PROMPT
            gen_context = self.update_context_text(system_prompt, gen_context)
            # cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            step = 0
            while step < max_steps:
                # if step > 0:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                # else:
                #     gen_text = "<think> generate an image of a mario </think>"
                gen_context = self.update_context_text(gen_text, gen_context)
                cfg_img_context = self.update_context_text(gen_text, cfg_img_context)
                output_list.append(gen_text)
                print(gen_text)
                if gen_text.strip().endswith("STOP"):
                    break

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
                output_list.append(img)
                gen_context = self.update_context_image(img, gen_context, vae=True, vit=True)
                # cfg_text_context = deepcopy(gen_context)
                cfg_text_context = self.update_context_image(img, cfg_text_context, vae=True, vit=True)
                step += 1

        return output_list

    def interleave_inference_multi_instructions(
        self,
        input_lists: List[Union[str, Image.Image]],
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        see_history=True
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        text_instruction_list = []

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    text_instruction_list.append(input_term)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            # Do generation for each text instruction, can see all the previous images and text instructions
            if see_history:
                for text_instruction in text_instruction_list:
                    # First update the context with the text instruction
                    print(text_instruction)
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(text_instruction, gen_context)
                    cfg_img_context = self.update_context_text(text_instruction, cfg_img_context)

                    # Then generate the image, we don't consider generating text, each step the text is merely from the instruction list
                    img = self.gen_image(
                        image_shapes, 
                        gen_context, 
                        cfg_text_precontext=cfg_text_context, 
                        cfg_img_precontext=cfg_img_context,

                        cfg_text_scale=cfg_text_scale, 
                        cfg_img_scale=cfg_img_scale, 
                        cfg_interval=cfg_interval, 
                        timestep_shift=timestep_shift, 
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    output_list.append(img)
                    
                    # Then update the context from the image
                    gen_context = self.update_context_image(img, gen_context, vae=True, vit=True)
                    # cfg_text_context = deepcopy(gen_context)
                    cfg_text_context = self.update_context_image(img, cfg_text_context, vae=True, vit=True)
            else:
                for text_instruction in text_instruction_list:
                    # update the context with the text instruction
                    print(text_instruction)
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(text_instruction, gen_context)
                    cfg_img_context = self.update_context_text(text_instruction, cfg_img_context)              
                     # Then generate the image, we don't consider generating text, each step the text is merely from the instruction list
                    img = self.gen_image(
                        image_shapes, 
                        gen_context, 
                        cfg_text_precontext=cfg_text_context, 
                        cfg_img_precontext=cfg_img_context,

                        cfg_text_scale=cfg_text_scale, 
                        cfg_img_scale=cfg_img_scale, 
                        cfg_interval=cfg_interval, 
                        timestep_shift=timestep_shift, 
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    output_list.append(img)
                    # Then we clear the context and only use the new image to update the context
                    # Clear context
                    gen_context = self.init_gen_context()
                    cfg_text_context = deepcopy(gen_context)
                    cfg_img_context = deepcopy(gen_context)
                    
                    # Use the final step image to update the context
                    gen_context = self.update_context_image(img, gen_context, vae=True, vit=True)

        return output_list

    def interleave_inference_multi_instructions_reflection(
        self,
        input_images: List[Image.Image],
        input_texts: List[str],
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        max_steps=6,
        save_dir="./results/reflection_multi_turn_results",
    ) -> List[Union[str, Image.Image]]:
        """
        Input: a list of image text sequence (I, T, I, T, ...)
        Output: each step, a text reflection is generated based on the sequence; then a new image is generated to fix this.
        """
        
        i = 0
        while os.path.exists(f"{save_dir}/sample_{i}"):
            i += 1
        save_dir = f"{save_dir}/sample_{i}"
        os.makedirs(save_dir, exist_ok=True) 
        output_list = []
        input_image_lists = input_images.copy()
        input_text_lists = input_texts.copy()
        system_prompt = REFLECTION_SYSTEM_PROMPT_MULTI_TURN
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            step = 0
            while step < max_steps:
                # first generate the reflection text + action text
                reflection_text_list = self.concat_text_prompts(input_text_lists, system_prompt)
                gen_context_reflection, _, _ = self.get_gen_context(input_image_lists=input_image_lists, input_text_lists=[reflection_text_list], understanding_output=True)
                gen_text = self.gen_text(gen_context_reflection, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)
                # save the text
                with open(f"{save_dir}/text_{step}.txt", "w") as f:
                    f.write(gen_text)
                if "</think>" in gen_text:
                    gen_text = gen_text.split("</think>\n")[1]
                input_text_lists.append(gen_text)
                print(gen_text)
                if "STOP" in gen_text.strip():
                    break
                # then generate the next image
                gen_context_single_step, cfg_text_context_single_step, cfg_img_context_single_step = self.get_gen_context(input_image_lists=input_image_lists, input_text_lists=input_text_lists[-1:], understanding_output=False)
                img = self.gen_image(
                    image_shapes, 
                    gen_context_single_step, 
                    cfg_text_precontext=cfg_text_context_single_step, 
                    cfg_img_precontext=cfg_img_context_single_step,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
                input_image_lists.append(img)
                img.save(f"{save_dir}/img_{step}.png")
                output_list.append(img)
                step += 1
        return output_list

    @torch.no_grad()
    def interleave_inference_reflection_prev(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        max_steps=4,
        single_step_image=False,
        single_step_text=False,
        save_dir="./results/reflection_results",
    ) -> List[Union[str, Image.Image]]:
        import os
        os.makedirs(save_dir, exist_ok=True)
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        gen_context_reflection = deepcopy(gen_context)


        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

            system_prompt = REFLECTION_THINK_SYSTEM_PROMPT

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")
            if single_step_image:
                gen_context_single_step = deepcopy(gen_context)
                cfg_text_context_single_step = deepcopy(cfg_text_context)
                cfg_img_context_single_step = deepcopy(cfg_img_context)
            if single_step_text:
                gen_context_text_single_step = deepcopy(gen_context)
            step = 0
            while step < max_steps:
                if not single_step_image:
                    img = self.gen_image(
                        image_shapes, 
                        gen_context, 
                        cfg_text_precontext=cfg_text_context, 
                        cfg_img_precontext=cfg_img_context,

                        cfg_text_scale=cfg_text_scale, 
                        cfg_img_scale=cfg_img_scale, 
                        cfg_interval=cfg_interval, 
                        timestep_shift=timestep_shift, 
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    img.save(f"{save_dir}/img_{step}.png")
                    output_list.append(img)
                    gen_context = self.update_context_image(img, gen_context, vae=True, vit=True)
                    cfg_text_context = self.update_context_image(img, cfg_text_context, vae=True, vit=True)
                    # copy the context for reflection
                    if not single_step_text:
                        gen_context_reflection = deepcopy(gen_context)
                        gen_context_reflection = self.update_context_text(system_prompt, gen_context_reflection)
                        gen_text = self.gen_text(gen_context_reflection, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    else:
                        gen_context_reflection = deepcopy(gen_context_text_single_step)
                        gen_context_reflection = self.update_context_image(img, gen_context_reflection, vae=False, vit=True)
                        gen_context_reflection = self.update_context_text(system_prompt, gen_context_reflection)
                        gen_text = self.gen_text(gen_context_reflection, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    cfg_img_context = self.update_context_text(gen_text, cfg_img_context)
                    output_list.append(gen_text)
                    print(gen_text)
                    # save the text
                    with open(f"{save_dir}/text_{step}.txt", "w") as f:
                        f.write(gen_text)
                    if gen_text.strip().endswith("STOP"):
                        break
                    step += 1
                else:
                    img = self.gen_image(
                        image_shapes, 
                        gen_context_single_step, 
                        cfg_text_precontext=cfg_text_context_single_step, 
                        cfg_img_precontext=cfg_img_context_single_step,

                        cfg_text_scale=cfg_text_scale, 
                        cfg_img_scale=cfg_img_scale, 
                        cfg_interval=cfg_interval, 
                        timestep_shift=timestep_shift, 
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(img))
                    gen_context_single_step = self.init_gen_context()
                    cfg_img_context_single_step = deepcopy(gen_context_single_step)
                    gen_context_single_step = self.update_context_image(input_term, gen_context_single_step, vae=True, vit=True)
                    cfg_text_context_single_step = deepcopy(gen_context_single_step)
                    img.save(f"{save_dir}/img_{step}.png")
                    output_list.append(img)
                    

                    if not single_step_text:
                    # copy the context for reflection
                        gen_context = self.update_context_image(input_term, gen_context, vae=False, vit=True)
                        cfg_text_context = self.update_context_image(input_term, cfg_text_context, vae=False, vit=True)
                        gen_context_reflection = deepcopy(gen_context)
                        gen_context_reflection = self.update_context_text(system_prompt, gen_context_reflection)
                        gen_text = self.gen_text(gen_context_reflection, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    else:
                        gen_context_reflection = deepcopy(gen_context_text_single_step)
                        gen_context_reflection = self.update_context_image(input_term, gen_context_reflection, vae=False, vit=True)
                        gen_context_reflection = self.update_context_text(system_prompt, gen_context_reflection)
                        gen_text = self.gen_text(gen_context_reflection, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)

                    output_list.append(gen_text)
                    # save the text
                    with open(f"{save_dir}/text_{step}.txt", "w") as f:
                        f.write(gen_text)
                    if "</think>" in gen_text:
                        gen_text = gen_text.split("</think>\n")[1]
                    if not single_step_text:
                        gen_context = self.update_context_text(gen_text, gen_context)
                        cfg_img_context = self.update_context_text(gen_text, cfg_img_context)
                    gen_context_single_step = self.update_context_text(gen_text, gen_context_single_step)
                    cfg_img_context_single_step = self.update_context_text(gen_text, cfg_img_context_single_step)
                    print(gen_text)
                    if gen_text.strip().endswith("STOP"):
                        break
                    step += 1
        return output_list

    def concat_text_prompts(self, text_lists: List[str], system_prompt: str):
        """
        Concatenate the text prompts into a single string
        """
        # return a single text prompt that will generate the next step text:
        prompt = "You are given an instruction and a list of actions you have taken to achieve the instruction. "
        prompt += "Also the images corresponding to the result after taking the actions are provided. "
        prompt += f"Instructions: {text_lists[0]} \n"
        prompt += f"Actions: {text_lists[1:]} \n"
        prompt += f"{system_prompt}"
        return prompt
    
    @torch.no_grad()
    def interleave_inference_reflection(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        max_steps=6,
        save_dir="./results/reflection_results",
    ) -> List[Union[str, Image.Image]]:
        import os
        # make a new sample_i dir under the save_dir, i should be the first sample that is not under save_dir
        i = 0
        while os.path.exists(f"{save_dir}/sample_{i}"):
            i += 1
        save_dir = f"{save_dir}/sample_{i}"
        os.makedirs(save_dir, exist_ok=True) 
        output_list = []
        input_image_lists = []
        input_text_lists = input_lists.copy()
        gen_context, cfg_text_context, cfg_img_context = self.get_gen_context(input_text_lists=input_text_lists, understanding_output=False)
        system_prompt = REFLECTION_THINK_SYSTEM_PROMPT
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):    
            step = 0
            while step < max_steps:
                # only use the last image to generate the next image
                # the first image should be generated through gen_context, cfg_text_context, cfg_img_context
                if step == 0:
                    gen_context_single_step, cfg_text_context_single_step, cfg_img_context_single_step = gen_context, cfg_text_context, cfg_img_context
                else:   
                    gen_context_single_step, cfg_text_context_single_step, cfg_img_context_single_step = self.get_gen_context(input_image_lists=input_image_lists[-1:], input_text_lists=input_text_lists[-1:], understanding_output=False)
                img = self.gen_image(
                    image_shapes, 
                    gen_context_single_step, 
                    cfg_text_precontext=cfg_text_context_single_step, 
                    cfg_img_precontext=cfg_img_context_single_step,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
                input_image_lists.append(img)
                img.save(f"{save_dir}/img_{step}.png")
                output_list.append(img)
                reflection_text_list = self.concat_text_prompts(input_text_lists, system_prompt)
                gen_context_reflection, _, _ = self.get_gen_context(input_image_lists=input_image_lists, input_text_lists=[reflection_text_list], understanding_output=True)
                gen_text = self.gen_text(gen_context_reflection, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)
                # save the text
                with open(f"{save_dir}/text_{step}.txt", "w") as f:
                    f.write(gen_text)
                if "</think>" in gen_text:
                    gen_text = gen_text.split("</think>\n")[1]
                input_text_lists.append(gen_text)
                print(gen_text)
                if "STOP" in gen_text.strip():
                    break
                step += 1
        return output_list
        
    @torch.no_grad()
    def interleave_action_with_logprobs(
        self,
        input_lists: List[Union[str, Image.Image]],
        max_new_tokens: int = 64,
        do_sample: bool = True,
        text_temperature: float = 1.0,
    ):
        """RL-rollout helper: build context from an interleaved [text|image]
        list, then emit ONE assistant text turn and return the text plus the
        per-token log-probs of the emitted tokens.

        Mirrors the understanding-output path of `interleave_inference` but
        replaces `gen_text` with `gen_text_with_logprobs`. Used by
        rl/rollout.py for GRPO. No image generation.

        Returns:
            text: str
            token_ids: list[int] of emitted (BOS/EOS-stripped) tokens
            logprobs: torch.Tensor(len(token_ids),) — log p(token | context) at T=1.0
        """
        gen_context = self.init_gen_context()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    gen_context = self.update_context_text(input_term, gen_context)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=False)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            text, token_ids, logprobs = self.gen_text_with_logprobs(
                gen_context,
                max_length=max_new_tokens,
                do_sample=do_sample,
                temperature=text_temperature,
            )
        return text, token_ids, logprobs

    @torch.no_grad()
    def get_gen_context(
        self,
        input_image_lists: List[Image.Image]=[],
        input_text_lists: List[str]=[],
        understanding_output=False,
    ):
        """
        Get the gen_context and cfg_text_context and cfg_img_context for the input_image_lists and input_text_lists
        """
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        # first update the gen_context and cfg_img_context with the input_image_lists
        for input_term in input_image_lists:
            input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
            gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)  
            cfg_img_context = self.update_context_image(input_term, cfg_img_context, vae=not understanding_output)
        cfg_text_context = deepcopy(gen_context)
        # then update the gen_context and cfg_img_context with the input_text_lists
        for input_term in input_text_lists:
            gen_context = self.update_context_text(input_term, gen_context)
            cfg_img_context = self.update_context_text(input_term, cfg_img_context)
        return gen_context, cfg_text_context, cfg_img_context

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        vae=None,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = pil_img2rgb(input_term)
                    if vae is None:
                        vae = not understanding_output
                    # Only pre-resize to VAE dims when the VAE branch will run.
                    # For understanding (vae=False), pass the raw PIL so vit_transform
                    # resizes directly — matching training, which applies vit_transform
                    # to the raw PIL once.
                    # if vae:
                    input_term = self.vae_transform.resize_transform(input_term)
                    gen_context = self.update_context_image(input_term, gen_context, vae=vae)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                print(gen_context)
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)

        return output_list

    
    def __call__(
        self, 
        input_image_list: Optional[List[Image.Image]] = None,
        input_text_list: Optional[List[str]] = None,
        interleaved_input: Optional[List[Union[str, Image.Image]]] = None,
        image: Optional[Image.Image] = None, # TODO: support multiple images
        text: Optional[Union[str, List[str]]] = None, # can be a list of str or a single str
        generation_option: Optional[str] = None,
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': [], 'text': []}

        input_list = []
        if interleaved_input is not None:
            input_list.extend(interleaved_input)
        if image is not None:
            input_list.append(image)
        if text is not None:
            if isinstance(text, str):
                input_list.append(text)
            elif isinstance(text, list):
                input_list.extend(text)
            else:
                raise ValueError(f"Unsupported input type: {type(text)}")
        if generation_option == "auto_trigger":
            output_list = self.interleave_inference_auto_trigger(input_list, **kargs)
        elif generation_option == "multi_turn":
            output_list = self.interleave_inference_multi_turn(input_list, **kargs)
        elif generation_option == "multi_instructions":
            output_list = self.interleave_inference_multi_instructions(input_list, **kargs)
        elif generation_option == "multi_instructions_reflection":
            output_list = self.interleave_inference_multi_instructions_reflection(input_image_list, input_text_list, **kargs)
        elif generation_option == "reflection":
            output_list = self.interleave_inference_reflection(input_list, **kargs)
        else:
            output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'].append(i)
            elif isinstance(i, str):
                output_dict['text'].append(i)
        return output_dict
    
    
    @torch.no_grad()
    def interleave_inference_auto_trigger(
        self,
        input_lists: list,
        max_steps: int = 512,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        **gen_kwargs,
    ) -> list:
        """
        Autoregressive loop that inspects each next token: if it's 'vision_start',
        calls gen_image(), otherwise decodes text.
        """
        outputs = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        # 1) Seed context with any initial inputs (prompt or images)
        vision_already_queued = False
        
        
        for input_term in input_lists:
            print(input_term)
            if isinstance(input_term, str):
                cfg_text_context = deepcopy(gen_context)
                gen_context = self.update_context_text(input_term, gen_context)
                cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                if input_term.strip().endswith("<|vision_start|>"):
                    vision_already_queued = True

            elif isinstance(input_term, Image.Image):
                input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                gen_context = self.update_context_image(input_term, gen_context, vae=True, vit=False)

                image_shapes = input_term.size[::-1]
                cfg_text_context = deepcopy(gen_context)

            else:
                raise ValueError(f"Unsupported input type: {type(input_term)}")
            
            print(gen_context)

        # 1a) If the prompt we just cached already ended with <|vision_start|>,
        #     call gen_image() right now (once), then skip straight to returning.
        if vision_already_queued:
            img = self.gen_image(
                image_shapes,
                gen_context, 
                cfg_text_precontext=cfg_text_context, 
                cfg_img_precontext=cfg_img_context,

                cfg_text_scale=cfg_text_scale, 
                cfg_img_scale=cfg_img_scale, 
                cfg_interval=cfg_interval, 
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            )
            outputs.append(img)

        # 2) Otherwise, run the usual generate_step loop
        prev_token = None
        for _ in range(max_steps):
            token_flag, new_past, new_kvlens, new_ropes, next_tokens = self.model.generate_step(
                gen_context['past_key_values'],
                gen_context['kv_lens'],
                gen_context['ropes'],
                self.new_token_ids,
                prev_token=prev_token,     
                do_sample=do_sample,
                temperature=text_temperature,
            )
            
            # update context
            gen_context['past_key_values'] = new_past
            gen_context['kv_lens']         = new_kvlens
            gen_context['ropes']           = new_ropes
            
            prev_token = next_tokens 

            # 3) Branch on token_flag
            if token_flag == 'vision_start':
                print(token_flag)
                img = self.gen_image(
                    image_shapes,
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
                outputs.append(img)
                input_term = self.vae_transform.resize_transform(pil_img2rgb(img))
                # input_term = input_term.to(torch.bfloat16)
                gen_context = self.update_context_image(input_term, gen_context, vae=True, vit=False)
                cfg_text_context = deepcopy(gen_context)
            else:
                text = self.tokenizer.decode([token_flag])
                outputs.append(text)
                print(text)
                # gen_context = self.update_context_text(text, gen_context)
                # cfg_text_context = deepcopy(gen_context)

            # 4) stop on EOS
            if token_flag == self.new_token_ids['eos_token_id']:
                print(token_flag)
                break

        return outputs
