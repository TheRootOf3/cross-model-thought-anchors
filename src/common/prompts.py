"""build_prefill_prompt: the exact string sent to /v1/completions for a model,
with A's reasoning-so-far prefilled inside the model's own reasoning block.

Everything about A's format is taken from the code that generated the dataset,
external/thought-anchors/generate_rollouts.py (@b53ed8c), not inferred:
  line 447/502 A's prompt is a RAW string, no chat template:
                TASK + "\\n<think>\\n" + prefix
  line 641-645 the prefix is " ".join(chunks).strip() (single spaces, not
                the original solution_text spacing)
  BOS the API's tokenizer prepended id 151646; ours does not, so
                a_raw_prompt starts with the literal A_BOS (2026-09-09)
Reader formats come from each model's chat_template.jinja rendered with
thinking enabled (see doc/setup.md, frozen by 01_prefill_probe.py).
"""

import re
from functools import lru_cache

from .io import REPO_ROOT, load_config

# generate_rollouts.py:447 - A's prompt up to the reasoning tag, byte-identical
# for every model (trailing space included); only the wrapping differs.
INSTRUCTION = ("Solve this math problem step by step. You MUST put your final answer "
               "in \\boxed{{}}. Problem: {problem} Solution: ")

# What each model's template leaves open after the assistant turn starts, and
# what must be appended before the prefix. Verified by rendering the templates
# (transformers, offline) on 2026-09-08; the probe re-verifies against vLLM.
_OPENER_AFTER_TEMPLATE = {
    "qwen3.5-9b": "",                                 # template ends "<think>\n"
    "qwen3.5-4b": "",
    "qwen3.5-2b": "",                                 # same family, same template
    "qwen3-4b-thinking-2507": "",                     # thinking-only; template ends "<think>\n"
    "qwen3.5-0.8b": "",                               # same family as the 2B
    "qwen3-1.7b": "<think>\n",                        # Qwen3 hybrid: template ends "assistant\n", model writes <think> itself
    "olmo3-7b-think": "",                             # template ends "<think>"
    "gpt-oss-20b": "<|channel|>analysis<|message|>",  # harmony analysis channel
}

# How each model opens and closes its reasoning block, and what counts as
# ABANDONING the prefilled one. Design rule: a reader that closes the block it
# was handed and starts again is no longer continuing A's reasoning-so-far,
# and the kept/replaced comparison is void for that rollout.
#
# The shape of the failure is template-specific, which is why one string match
# is not enough. The three `<think>` models can only re-open by emitting a
# fresh `<think>`. gpt-oss's harmony format has named channels, and it
# abandons the block by terminating it and opening a NEW assistant message in
# the `commentary` channel (15 of 16 observed re-opens) or in `analysis`
# (1 of 16) - while a transition to `final` is the normal path to the answer
# and must NOT count. Found 2026-09-08 by re-reading raw completions; the
# earlier analysis-only matcher saw 1 of the 16.
REASONING_MARKERS = {
    "r1-distill-14b": ("<think>", "</think>"),
    "qwen3.5-9b": ("<think>", "</think>"),
    "qwen3.5-4b": ("<think>", "</think>"),
    "qwen3.5-2b": ("<think>", "</think>"),
    "qwen3-4b-thinking-2507": ("<think>", "</think>"),
    "qwen3.5-0.8b": ("<think>", "</think>"),
    "qwen3-1.7b": ("<think>", "</think>"),
    "olmo3-7b-think": ("<think>", "</think>"),
    "gpt-oss-20b": ("<|start|>assistant<|channel|>analysis<|message|>", "<|end|>"),
}

# A new assistant message in any channel other than `final`.
_HARMONY_REOPEN = re.compile(
    r"<\|end\|><\|start\|>assistant<\|channel\|>(?!final\b)(\w+)<\|message\|>")
_HARMONY_FINAL = "<|channel|>final<|message|>"


def block_flags(model_key: str, text: str, finish_reason: str) -> dict:
    """Per-rollout flags describing what the reader did with the prefilled
    block. Derived from the raw completion, so it can be recomputed offline for
    rows already written - `reopened` in rows written before 2026-09-08 is
    stale for gpt-oss and must be re-derived, not trusted."""
    opener, closer = REASONING_MARKERS[model_key]

    if model_key == "gpt-oss-20b":
        m = _HARMONY_REOPEN.search(text)
        reopened, reopen_channel = bool(m), (m.group(1) if m else None)
        answer_at = text.find(_HARMONY_FINAL)
        reached_final = answer_at >= 0
    else:
        reopened = opener in text
        reopen_channel = "think" if reopened else None
        answer_at = text.find(closer)
        reached_final = answer_at >= 0 and bool(text[answer_at + len(closer):].strip())

    return {
        "reopened": reopened,
        "reopen_channel": reopen_channel,
        "closed": closer in text,
        "capped": finish_reason == "length",
        "reached_final": reached_final,
        # Reasoning = everything before the answer section, so a model that
        # does its derivation after the closer (gpt-oss does, ~18% of its
        # output) is not credited with less reasoning than it did.
        "reasoning_chars": answer_at if answer_at >= 0 else len(text),
        "post_answer_chars": (len(text) - answer_at) if answer_at >= 0 else 0,
    }


def join_prefix(sentences: list[str]) -> str:
    """generate_rollouts.py:641-645. Kept pile: sentences[:i+1]; replaced: [:i]."""
    return " ".join(sentences).strip()


# A's generations start with its BOS token. The dataset's rollouts were
# requested from a completions endpoint with the raw string
# (generate_rollouts.py:46 default provider Novita, :239 "prompt": prompt,
# :247 .../openai/completions); the model's tokenizer_config.json says
# add_bos_token true (LlamaTokenizerFast), which the 2025 serving stacks
# honoured. transformers 5.16.1 drops the config's add_bos_token whenever a
# tokenizer.json is present (tokenization_utils_base.py:1779-1781) and loads
# the model as Qwen2Tokenizer with add_bos_token False, so our vLLM 0.28 server
# adds none (/tokenize of a raw prompt has no id 151646).
# Hence the literal string, which the tokenizer maps
# to the single id 151646 (verified 2026-09-09, scripts/95_bos_ab.py). Human
# decision 2026-09-08 ("A runs with BOS"), made effective 2026-09-09.
A_BOS = "<｜begin▁of▁sentence｜>"


def a_raw_prompt(problem: str, prefix: str) -> str:
    """A's exact prompt, generate_rollouts.py:502, behind its BOS token (A_BOS).
    Used for the calibration run (reader = A)."""
    return A_BOS + INSTRUCTION.format(problem=problem) + "\n<think>\n" + prefix


@lru_cache(maxsize=None)
def _tokenizer(model_key: str):
    """Tokenizer only, never a model; offline; loaded once per key."""
    from transformers import AutoTokenizer

    hf_path = load_config(REPO_ROOT / "configs/models.json")["models"][model_key]["hf_path"]
    return AutoTokenizer.from_pretrained(hf_path, local_files_only=True)


PIN_RENDER_DATE: str | None = None      # set by a script; see pin_render_date()
_DATE_LINE = re.compile(r"(Current date: )\d{4}-\d{2}-\d{2}")


def render_date(prompt: str) -> str | None:
    """The date gpt-oss's harmony template stamped into a stored prompt, or None."""
    m = _DATE_LINE.search(prompt)
    return m.group(0).split(": ")[1] if m else None


def pin_render_date(date: str | None) -> None:
    """Freeze the date the chat template stamps into the system block.

    gpt-oss's harmony template renders `Current date: <today>` at call time
    (its `_reasoning_note` in configs/models.json: keep the template's
    defaults). A run that crosses midnight, resumes on another day, or reuses
    another run's piles would otherwise mix two prompts - and in the
    both-directions extension the matched piles (2026-09-09) and the new
    opposite piles would differ systematically, confounding the two sets.
    Scripts pin the date of the rows they extend; None = today's render.
    Found by a check, 2026-09-10."""
    global PIN_RENDER_DATE
    PIN_RENDER_DATE = date


def apply_pin(rendered: str) -> str:
    """Replace the template's call-time date with the pinned one, if a script pinned it.

    Read PIN_RENDER_DATE through this function, never by importing the name: `from ... import
    PIN_RENDER_DATE` binds the value at import time, so a later pin_render_date() would not reach
    the importing module and the pin would silently do nothing.
    """
    return _DATE_LINE.sub(r"\g<1>" + PIN_RENDER_DATE, rendered) if PIN_RENDER_DATE else rendered


def _chat_prefix(model_key: str, problem: str) -> str:
    """The model's own chat template rendered on the REAL instruction, with the
    assistant turn opened and thinking on, plus the opener the template leaves
    out. Rendered per prompt (~1 ms): rendering once with a placeholder and
    substituting afterwards bypasses what the template does to user content -
    Qwen3.5's `|trim` strips INSTRUCTION's trailing space, and the placeholder
    route put it back."""
    rendered = _tokenizer(model_key).apply_chat_template(
        [{"role": "user", "content": INSTRUCTION.format(problem=problem)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    rendered = apply_pin(rendered)
    return rendered + _OPENER_AFTER_TEMPLATE[model_key]


def build_prefill_prompt(model_key: str, problem: str, prefix_sentences: list[str]) -> str:
    prefix = join_prefix(prefix_sentences)
    if model_key == "r1-distill-14b":
        return a_raw_prompt(problem, prefix)
    return _chat_prefix(model_key, problem) + prefix
