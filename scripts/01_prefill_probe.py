#!/usr/bin/env python
"""Setup probe: does this reader CONTINUE a prefilled reasoning block?

    python scripts/01_prefill_probe.py --model qwen3.5-9b [--n 8] [--n_sentences 8]
        [--config configs/phase1.json] [--run_id probe_<model>]
        [--trace correct_base_solution/problem_1591]

One real A trace, its first `n_sentences` sentences prefilled, `n` continuations
at the design's reader sampling read from the config's `reader_sampling` block
(T 0.6, top_p 0.95, cap `max_tokens_math` = 16384 - the thought-anchors
generation cap; see the design changelog, 2026-09-08). Every continuation is
printed in full and written to runs/<run_id>/rows.jsonl as it lands, with:
  reopened - the model abandoned the block it was handed: a fresh <think>,
              or (harmony) a new assistant message in any channel but `final`
  reopen_channel - which channel it re-opened into, for the harmony models
  closed - the reasoning closer appears (the model finished reasoning)
  capped - hit max_tokens
  reached_final - the block closed and (gpt-oss) the final channel opened
  reasoning_chars / post_answer_chars - split at the answer section, so a
              model that derives after the closer is not credited with less
  answer / correct - first non-placeholder \\boxed{}, author's equivalence
Rerunning with the same --run_id never regenerates: it shows the rows on disk.
Only the canonical probe (default trace and n_sentences) freezes the prompt.
Before sampling it asserts the prompt is right: for readers, the server's own
chat-template render (/tokenize) must be an exact token prefix of our prompt;
for A, our prompt must equal the dataset's stored prompt at that position.
The exact prompt string is frozen into doc/setup.md. Read the output.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.answers import extract_answer, is_correct
from src.common.client import VLLMClient
from src.common.data import ROLLOUTS_ROOT, Trace
from src.common.io import REPO_ROOT, append_row, load_config, read_rows, rng, run_dir, write_run_config
from src.common.prompts import A_BOS, INSTRUCTION, block_flags, build_prefill_prompt

A_KEY = "r1-distill-14b"
DEFAULT_TRACE, DEFAULT_N_SENTENCES = "correct_base_solution/problem_1591", 8


def assert_prompt_correct(client, model, trace, n_sentences, prompt):
    """The model's own template, rendered by the server, must be a token
    prefix of what we send; A's raw prompt must be the dataset's own."""
    if model == A_KEY:
        assert n_sentences < trace.n_chunks, f"--n_sentences {n_sentences} >= {trace.n_chunks} chunks"
        stored = trace.rollouts(n_sentences)[0]
        stored_prompt = stored["full_cot"][: len(stored["full_cot"]) - len(stored["rollout"])]
        if not trace.prefix_matches_dataset(n_sentences):
            head = INSTRUCTION.format(problem=trace.question) + "\n<think>\n"
            assert prompt.startswith(A_BOS + head) and stored_prompt.startswith(head)
            return (f"WARNING chunk {n_sentences} is a corrupted-prefix position (generator's "
                    f"str.replace bug); only the instruction head was compared")
        assert prompt == A_BOS + stored_prompt, "A prompt != BOS + dataset's stored prompt"   # BOS: prompts.py A_BOS, 2026-09-09
        return f"A prompt == BOS + dataset's stored prompt at chunk {n_sentences}"
    chat = client.tokenize_chat([{"role": "user", "content": INSTRUCTION.format(problem=trace.question)}])
    ours = client.tokenize_text(prompt)
    assert ours[: len(chat)] == chat, (
        f"server template render is NOT a prefix of our prompt; first diff at "
        f"{next(i for i, (a, b) in enumerate(zip(chat, ours)) if a != b)}")
    return f"server chat render ({len(chat)} tokens) is a prefix of our prompt ({len(ours)} tokens)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n_sentences", type=int, default=DEFAULT_N_SENTENCES)
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--sampling", choices=["phase1", "recommended"], default="phase1",
                    help="phase1 = the pre-registered design reader sampling from --config; recommended = the model's own "
                         "settings from configs/models.json (experiment 2026-09-08)")
    ap.add_argument("--timeout", type=float, default=900.0, help="client read timeout, s (n=32 x 32k needs more)")
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="override the config's cap (experiment 2026-09-08: 32768 = 2x the pre-registered design)")
    args = ap.parse_args()
    run_id = args.run_id or f"probe_{args.model}"

    cfg = load_config(args.config)
    # Two sampling conditions - the contrast the setup figure (fig0a) is built on:
    # recommended = the reader's own published settings, which every later phase uses;
    # phase1 = A's sampler (T 0.6 / top-p 0.95), the settings A's released rollouts were
    # generated with, applied to the reader instead.
    key = args.model if args.sampling == "recommended" else "r1-distill-14b"
    samp = {k: v for k, v in load_config(REPO_ROOT / "configs/models.json")["models"][key]
            ["recommended_sampling"].items() if not k.startswith("_")}
    samp["max_tokens_math"] = cfg["reader_sampling"]["max_tokens_math"]
    if args.max_tokens:
        samp["max_tokens_math"] = args.max_tokens
    trace = Trace(ROLLOUTS_ROOT / args.trace, args.trace.split("/")[0])
    rows_path = run_dir(run_id) / "rows.jsonl"

    rows = list(read_rows(rows_path))
    if rows:  # resume = never regenerate; show what is on disk
        prompt, check = rows[0]["prompt"], "(resumed from rows.jsonl)"
        print(f"{rows_path.relative_to(REPO_ROOT)} already has {len(rows)} rows; not sampling again. "
              "Use a new --run_id for a fresh probe.")
    else:
        client = VLLMClient(args.model, timeout=args.timeout)
        client.assert_serving()
        prompt = build_prefill_prompt(args.model, trace.question, trace.chunks[: args.n_sentences])
        check = assert_prompt_correct(client, args.model, trace, args.n_sentences, prompt)
        seed = rng(cfg["seed"], "probe", args.model, args.trace, args.n_sentences).getrandbits(31)
        write_run_config(run_id, cfg, extra={"script": "01_prefill_probe", "args": vars(args),
                                             "sampling_used": samp, "seed": seed, "prompt": prompt,
                                             "prompt_check": check})
        extra = {k: v for k, v in samp.items() if k not in ("temperature", "top_p", "max_tokens_math", "max_tokens_mmlu")}
        for i, c in enumerate(client.complete(prompt, args.n, samp["max_tokens_math"],
                                              samp["temperature"], samp["top_p"], seed=seed, **extra)):
            answer = extract_answer(c["text"])
            row = {"run_id": run_id, "phase": "probe", "model": args.model, "trace": args.trace,
                   "n_sentences": args.n_sentences, "sample_idx": i, "seed": seed, "prompt": prompt,
                   "completion": c["text"], "finish_reason": c["finish_reason"], "tokens": c["tokens"],
                   **block_flags(args.model, c["text"], c["finish_reason"]),
                   "answer": answer, "correct": is_correct(answer, trace.gt_answer),
                   "ts": dt.datetime.now(dt.UTC).isoformat()}
            append_row(rows_path, row)
            rows.append(row)

    print(f"model {args.model} | trace {args.trace} | {args.n_sentences} sentences prefilled | n={len(rows)} "
          f"| sampling={args.sampling} {({k: v for k, v in samp.items() if 'max_tokens' not in k})} | cap={samp['max_tokens_math']}")
    print(f"prompt check: {check}\nprompt tail: {prompt[-300:]!r}\n")
    for r in rows:
        flags = " ".join(k for k in ("reopened", "closed", "capped", "reached_final") if r.get(k))
        print(f"{'=' * 78}\n[{r['sample_idx']}] tokens={r['tokens']} finish={r['finish_reason']} "
              f"answer={r['answer']!r} correct={r['correct']} reasoning_chars={r.get('reasoning_chars')} "
              f"{flags}\n{'-' * 78}\n{r['completion']}")

    # .get: rows written before reached_final existed lack it
    summary = {k: sum(bool(r.get(k)) for r in rows) for k in ("reopened", "closed", "reached_final", "capped", "correct")}
    summary["with_answer"] = sum(r["answer"] is not None for r in rows)
    print(f"\n{'=' * 78}\nSUMMARY {args.model}: n={len(rows)} " + " ".join(f"{k}={v}" for k, v in summary.items()))
    print(f"gt_answer={trace.gt_answer!r}  rows -> {rows_path.relative_to(REPO_ROOT)}")

    if (args.trace, args.n_sentences, args.sampling, args.max_tokens) == (DEFAULT_TRACE, DEFAULT_N_SENTENCES, "phase1", None):
        freeze_in_doc(args.model, args.trace, args.n_sentences, prompt, check, summary, len(rows))
    else:
        print("non-default trace/n_sentences: prompt NOT frozen into doc/setup.md (canonical probe only)")
    print("\nCHECKPOINT: read every continuation above. Did the model continue A's "
          "reasoning inside the block it was handed? (reopened=0 is necessary, not sufficient.)")


def freeze_in_doc(model, trace_name, n_sentences, prompt, check, summary, n):
    """Store the exact working prompt string for this model."""
    doc = REPO_ROOT / "doc/setup.md"
    begin, end = f"<!-- BEGIN probe {model} -->", f"<!-- END probe {model} -->"
    block = (f"{begin}\n### Prefill prompt, `{model}` (frozen by `01_prefill_probe.py`, "
             f"{dt.date.today()})\n\n{check}. Trace `{trace_name}`, {n_sentences} sentences; "
             f"n={n}: " + ", ".join(f"{k}={v}" for k, v in summary.items())
             + f"\n\n```python\n{prompt!r}\n```\n{end}")
    text = doc.read_text() if doc.exists() else "# Setup\n"
    if begin in text:
        text = text[: text.index(begin)] + block + text[text.index(end) + len(end):]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    doc.write_text(text)
    print(f"frozen prompt -> doc/setup.md ({begin})")


if __name__ == "__main__":
    main()
