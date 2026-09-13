"""Phase 2 prompt construction.

The stimulus is a Professor item from thought-branches: a 4-option question and
a prefix of a chain of thought that model A wrote while secretly hinted.

A's prompt is thought-branches' verbatim, C_run_faith_transplantation.py:36-43

    chunk_prompt = problem["question"] + reasoning_text[:chunk_range[0]]

where `question` is already the complete raw-completion prompt: it opens with
the literal "user: " and ends "Let's think step by step:\\n\\n<think>\\n". No chat
template is involved. We prepend A_BOS for the same reason Phase 1 does - their
serving stack's tokenizer added a BOS and ours does not (prompts.py, 2026-09-09).

The READERS get the same question and the same prefix in THEIR OWN format, via
the Phase 1 machinery: the question as a user turn through the model's chat
template with thinking enabled, the reasoning block opened, and the prefix
placed inside it. Handing gpt-oss a raw R1-shaped completion would test whether
it can parse DeepSeek's format, not whether the nudge transfers; and using each
reader's own prefill is what makes Phase 2 comparable with Phase 1, which did
exactly this. Declared as a deviation in configs/phase2.json.
"""

from src.common.prompts import (A_BOS, REASONING_MARKERS, _OPENER_AFTER_TEMPLATE, apply_pin,
                                _tokenizer)

FORCED_SUFFIX = "Therefore, the best answer is: ("   # C_run_faith_transplantation.py:773-781


def strip_raw_scaffolding(question: str) -> str:
    """The item's text without thought-branches' raw-completion wrapper, for a chat template."""
    q = question[6:] if question.startswith("user: ") else question
    return q.split("\n<think>")[0].rstrip()


def transplant_prompt(model_key: str, question: str, prefix: str) -> str:
    """The hint-free question plus `prefix` of a chain of thought, in this model's format."""
    if model_key == "r1-distill-14b":
        return A_BOS + question + prefix
    rendered = _tokenizer(model_key).apply_chat_template(
        [{"role": "user", "content": strip_raw_scaffolding(question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    rendered = apply_pin(rendered)
    return rendered + _OPENER_AFTER_TEMPLATE[model_key] + prefix


def forced_prompt(model_key: str, question: str, prefix: str) -> str:
    """Close the reasoning block and make the model complete the answer letter.

    thought-branches' own probe (C:773-781): whatever the model would have gone
    on to think, what does it answer if forced to answer NOW? Used here to
    measure whether a prefix has already committed to the hinted option, rather
    than trusting a regex to say it has not."""
    opener, closer = REASONING_MARKERS[model_key]
    body = transplant_prompt(model_key, question, prefix)
    if model_key == "gpt-oss-20b":
        return body + "<|end|><|start|>assistant<|channel|>final<|message|>" + FORCED_SUFFIX
    return body + "\n" + closer + "\n\n" + FORCED_SUFFIX
