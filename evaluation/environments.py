# Modified for Agima: package imports and portable resource configuration.
"""
Generalization-eval wrappers for VLM-Gym environments.

ActionNameRemapEnv
    Rename action function names in prompts at test time, then remap
    the model's renamed actions back to the original names before
    calling env.step().  Zero changes to the model or the env.

Sim2RealEnv
    Intercept env.render() and pass each frame through a style-transfer
    callable to make sim images look like real photographs.
"""

import inspect
import re
from typing import Dict, Optional, Callable

import numpy as np
from PIL import Image


def _mark_wrapper_reset(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_has_reset"):
            current._has_reset = True
        inner = getattr(current, "_env", None)
        if inner is None:
            inner = getattr(current, "env", None)
        current = inner


def _reset_accepts_init_state(env) -> bool:
    reset = getattr(env, "reset", None)
    if reset is None:
        return False
    try:
        sig = inspect.signature(reset)
    except (TypeError, ValueError):
        return False

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "init_state" in sig.parameters


def _reset_preserving_init_state(env, **kwargs):
    """Forward init_state through wrapper stacks that hide custom reset kwargs."""
    if kwargs.get("init_state", None) is None:
        kwargs = {k: v for k, v in kwargs.items() if k != "init_state"}

    try:
        return env.reset(**kwargs)
    except TypeError as exc:
        if "init_state" not in kwargs or "unexpected keyword argument 'init_state'" not in str(exc):
            raise

    init_state = kwargs["init_state"]
    base_kwargs = {k: v for k, v in kwargs.items() if k != "init_state"}

    unwrapped = getattr(env, "unwrapped", None)
    if (
        unwrapped is not None
        and unwrapped is not env
        and _reset_accepts_init_state(unwrapped)
    ):
        try:
            result = unwrapped.reset(init_state=init_state, **base_kwargs)
            _mark_wrapper_reset(env)
            return result
        except TypeError as unwrapped_exc:
            if "unexpected keyword argument 'init_state'" not in str(unwrapped_exc):
                raise

    current = env
    seen = {id(env)}
    traversed = [env]

    while True:
        inner = getattr(current, "_env", None)
        if inner is None:
            inner = getattr(current, "env", None)
        if inner is None or id(inner) in seen:
            break
        seen.add(id(inner))
        traversed.append(inner)
        current = inner
        if not _reset_accepts_init_state(inner):
            continue
        try:
            result = inner.reset(init_state=init_state, **base_kwargs)
            for wrapped in traversed:
                if hasattr(wrapped, "_has_reset"):
                    wrapped._has_reset = True
            _mark_wrapper_reset(env)
            return result
        except TypeError as inner_exc:
            if "unexpected keyword argument 'init_state'" not in str(inner_exc):
                raise

    return env.reset(**base_kwargs)


# ---------------------------------------------------------------------------
# Action-name remap wrapper
# ---------------------------------------------------------------------------

class ActionNameRemapEnv:
    """
    Wraps any VLM-Gym env to test action-name generalization.

    name_map: dict  {new_name -> original_name}
        e.g. {"exchange": "swap", "shuffle": "reorder"}

    What it does
    ------------
    * get_prompt()  – replaces every occurrence of original_name with
                      new_name so the VLM is trained to use the new vocabulary.
    * step(action)  – the VLM will output ('exchange', ...) etc.; this
                      remaps those names back to originals before calling
                      the real env.step(), which only knows the originals.
    """

    def __init__(self, env, name_map: Dict[str, str]):
        self._env = env  # use _env (not .env) so Interactor's unwrap loop doesn't strip this wrapper
        # new_name -> original_name
        self.name_map = name_map
        # original_name -> new_name  (for prompt substitution)
        self.inv_map = {v: k for k, v in name_map.items()}

    # ------------------------------------------------------------------ #
    # Prompt: replace original names → new names                         #
    # ------------------------------------------------------------------ #
    def get_prompt(self, **kwargs) -> str:
        # Use .unwrapped to skip gym wrappers (OrderEnforcing, TimeLimit) that don't have get_prompt
        inner = self._env.unwrapped if hasattr(self._env, 'unwrapped') else self._env
        prompt = inner.get_prompt(**kwargs)
        for orig, new in self.inv_map.items():
            # Replace single-quoted occurrences:  'swap' → 'exchange'
            prompt = prompt.replace(f"'{orig}'", f"'{new}'")
            # Replace double-quoted occurrences:  "swap" → "exchange"
            prompt = prompt.replace(f'"{orig}"', f'"{new}"')
            # Replace bare-word occurrences in prose / format strings,
            # but only whole words to avoid partial matches.
            prompt = re.sub(rf'\b{re.escape(orig)}\b', new, prompt)
        print(f"Prompt: {prompt}")
        return prompt

    # ------------------------------------------------------------------ #
    # Step: remap new names → original names before forwarding           #
    # ------------------------------------------------------------------ #
    def step(self, action: str):
        remapped = action
        for new, orig in self.name_map.items():
            # Only replace the quoted first-element, e.g. 'exchange' → 'swap'
            remapped = remapped.replace(f"'{orig}'", f"'{new}'")
            remapped = remapped.replace(f'"{orig}"', f'"{new}"')
        return self._env.step(remapped)

    # ------------------------------------------------------------------ #
    # Forward everything else to the wrapped env                         #
    # ------------------------------------------------------------------ #
    def reset(self, **kwargs):
        return _reset_preserving_init_state(self._env, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' has no attribute 'env'")
        return getattr(self._env, name)

    def __str__(self):
        return f"ActionNameRemapEnv({self._env}, name_map={self.name_map})"

    def __repr__(self):
        return self.__str__()


# ---------------------------------------------------------------------------
# Sim2Real render wrapper
# ---------------------------------------------------------------------------

class Sim2RealEnv:
    """
    Wraps any VLM-Gym env and passes rendered frames through a
    style-transfer callable before they reach the VLM.

    style_fn: Callable[[np.ndarray], np.ndarray | PIL.Image]
        Takes a rendered sim frame (H×W×3 uint8), returns a PIL Image
        or numpy array of a realistic-looking version.
        Pass None to use identity (no transformation).
    """

    def __init__(self, env, style_fn: Optional[Callable] = None):
        self._env = env  # use _env (not .env) so Interactor's unwrap loop doesn't strip this wrapper
        self.style_fn = style_fn
        self._first_render_done = False

    def render(self, *args, **kwargs):
        frame = self._env.render(*args, **kwargs)
        # Only apply style transfer on the very first render of each episode
        if self.style_fn is not None and not self._first_render_done:
            result = self.style_fn(frame)
            if isinstance(result, Image.Image):
                result = np.array(result.convert("RGB"))
            frame = result
            self._first_render_done = True
        return frame

    def reset(self, **kwargs):
        self._first_render_done = False
        return _reset_preserving_init_state(self._env, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def get_prompt(self, **kwargs):
        inner = self._env.unwrapped if hasattr(self._env, 'unwrapped') else self._env
        return inner.get_prompt(**kwargs)

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' has no attribute 'env'")
        return getattr(self._env, name)

    def __str__(self):
        return f"Sim2RealEnv({self._env})"

    def __repr__(self):
        return self.__str__()


# ---------------------------------------------------------------------------
# Style-transfer factory functions
# ---------------------------------------------------------------------------

def _pil_to_data_url(im: Image.Image, fmt: str = "PNG") -> str:
    import io as _io, base64 as _b64
    buf = _io.BytesIO()
    im.convert("RGB").save(buf, format=fmt)
    b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def _data_url_to_pil(data_url: str) -> Image.Image:
    import io as _io, base64 as _b64
    _, encoded = data_url.split(",", 1)
    return Image.open(_io.BytesIO(_b64.b64decode(encoded))).convert("RGB")


def _b64_to_pil(b64_str: str) -> Image.Image:
    import io as _io, base64 as _b64
    return Image.open(_io.BytesIO(_b64.b64decode(b64_str))).convert("RGB")


def _parse_nanobana_response(result: dict) -> Image.Image:
    """Extract the first image from an OpenRouter / Gemini chat response."""
    msg = result["choices"][0]["message"]
    for img in msg.get("images", []):
        if "b64_json" in img:
            return _b64_to_pil(img["b64_json"])
        if "image_url" in img and "url" in img["image_url"]:
            return _data_url_to_pil(img["image_url"]["url"])
    raise RuntimeError(f"No image found in response: {str(result)[:500]}")


NANOBANA_MODEL = "google/gemini-3.1-flash-image-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SIM2REAL_PROMPT = (
    "Convert this simulation image into a photorealistic real-world image while preserving "
    "the scene exactly. Keep identical camera pose, framing, geometry, scale, proportions, "
    "robot configuration, object placement, and all spatial relationships. "
    "Do not alter composition in any way. Do not move or resize anything. "
    "Only replace the simulated appearance with realistic 3D materials, lighting, shadows, "
    "textures, and real lab photo style."
)

SIM2REAL_CUBE_PROMPT = (
    "This image shows a side-by-side pair of 3D objects built from unit cubes, "
    "rendered in a minimalist simulation style on a plain white background. "
    "Convert both objects into photorealistic 3D renders: add realistic materials "
    "(e.g. matte plastic or wood-like texture), natural soft lighting with subtle "
    "shadows, and slight depth-of-field. "
    "Preserve the exact shape, size, orientation, and spatial layout of both objects "
    "with pixel-level fidelity. "
    "Do not change the camera angle, composition, or object geometry in any way. "
    "Keep the background clean white."
)

# Map preset name → prompt string
SIM2REAL_PRESETS: Dict[str, str] = {
    "mujoco": SIM2REAL_PROMPT,
    "cube":   SIM2REAL_CUBE_PROMPT,
}

SIM2REAL_ICL_PROMPT = (
    "You are given three images:\n\n"
    "Image 1: a simulation image.\n"
    "Image 2: the corresponding real-world version of Image 1, showing the desired visual style.\n"
    "Image 3: a new simulation image to convert.\n\n"
    "Task:\nConvert Image 3 into the same real-world visual style as Image 2.\n\n"
    "Important constraints:\n"
    "- Use Image 2 only as a style reference.\n"
    "- Preserve Image 3's camera viewpoint, framing, geometry, object sizes, proportions,\n"
    "  robot pose, object placement, and all spatial relationships exactly.\n"
    "- Do not copy the scene content or layout from Image 2.\n"
    "- Do not move, resize, rotate, add, or remove anything in Image 3.\n"
    "- Keep Image 3 pixel-aligned in composition as much as possible.\n"
    "- Only change appearance: materials, texture, lighting, shadows, reflections,\n"
    "  background realism, and photographic lab style.\n\n"
    "Output:\nA photorealistic real-world version of Image 3 with the same style as Image 2,\n"
    "while preserving Image 3's structure exactly."
)


def make_nanobana_sim2real_fn(
    prompt: str = SIM2REAL_PROMPT,
    api_key: Optional[str] = None,
    model: str = NANOBANA_MODEL,
):
    """
    Returns a style_fn that calls Gemini via OpenRouter (NanoBanana) for sim→real.
    Requires OPENROUTER_API_KEY in environment if api_key is not passed.
    """
    import os, requests as _req

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    assert key, "Set OPENROUTER_API_KEY env var for NanoBanana sim2real"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def nanobana_style_fn(frame: np.ndarray) -> Image.Image:
        img = Image.fromarray(frame).convert("RGB")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}},
            ]}],
            "modalities": ["image", "text"],
        }
        resp = _req.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        return _parse_nanobana_response(resp.json())

    return nanobana_style_fn


def make_nanobana_sim2real_icl_fn(
    example_sim: Image.Image,
    example_real: Image.Image,
    prompt: str = SIM2REAL_ICL_PROMPT,
    api_key: Optional[str] = None,
    model: str = NANOBANA_MODEL,
):
    """
    Returns a style_fn that uses a (sim, real) example pair as style reference
    (in-context learning mode).
    """
    import os, requests as _req

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    assert key, "Set OPENROUTER_API_KEY env var for NanoBanana sim2real"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    sim_url  = _pil_to_data_url(example_sim)
    real_url = _pil_to_data_url(example_real)

    def nanobana_icl_style_fn(frame: np.ndarray) -> Image.Image:
        img = Image.fromarray(frame).convert("RGB")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": sim_url}},
                {"type": "image_url", "image_url": {"url": real_url}},
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}},
            ]}],
            "modalities": ["image", "text"],
        }
        resp = _req.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        return _parse_nanobana_response(resp.json())

    return nanobana_icl_style_fn


def make_identity_style_fn():
    """No-op style function — useful as ablation baseline."""
    def identity(frame: np.ndarray) -> np.ndarray:
        return frame
    return identity


# ---------------------------------------------------------------------------
# Pre-defined action-name presets
# (only ref_dot and colorization are intended targets; jigsaw not included)
# ---------------------------------------------------------------------------

#: Ready-made name maps — pass one of these as the name_map arg.
#:
#: ref_dot:
#:   'mark'     → 'click'   (pointing as clicking, more natural)
#:
#: colorization:
#:   'rotate'   → 'hue_shift'              (describes the actual change)
#:   'saturate' → 'adjust_saturation'      (more descriptive alternative)
ACTION_PRESETS: Dict[str, Dict[str, str]] = {
    # --- ref_dot/RefCOCOPlusDotEnv-v0 ---
    "ref_dot_click": {
        "click": "mark",       # new_name -> original_name
    },
    # --- colorization/ColorizationEnv-v2 ---
    "colorization_rename": {
        "hue_shift":          "rotate",
        "adjust_saturation":  "saturate",
    },
    # Combined: both envs at once (safe — keys don't overlap)
    "ref_dot_and_colorization": {
        "click":              "mark",
        "hue_shift":          "rotate",
        "adjust_saturation":  "saturate",
    },
}
