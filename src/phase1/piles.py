"""Kept and replaced piles for one (reader, trace, sentence i): the Phase 1
measurement (reader importance).

kept prefill A's sentences 1..i (chunks[:i+1]); the reader continues.
replaced prefill 1..i-1 (chunks[:i]); the reader writes its own sentence i and
          continues. Its first sentence (src/common/split.first_sentence, the
          same splitter the dataset used) is the replacement; the rollout
          survives the semantic filter iff cosine(replacement, A's sentence i)
          < 0.8 (src/common/filters).
Rows are written one per rollout as they land (io.append_row); a rerun
resumes per (trace, sentence, pile). A rollout is USABLE for P(correct) iff
it did not re-open the reasoning block (the pre-registration) and, in the replaced pile,
survived the filter. P(correct) is answered-only (answers.p_correct).
Top-ups (configs/phase1.json importance): the replaced pile gets
`replaced_topup_samples` more once if fewer than `replaced_min_survivors`
survive; either pile gets `answered_topup_samples` more once if fewer than
`min_answered_per_pile` usable rollouts answered. Every top-up is a row
field, never silent.
"""

import datetime as dt
import json
from pathlib import Path

from src.common.answers import extract_answer, is_correct, p_correct
from src.common.io import append_row, read_rows, rng
from src.common.prompts import REASONING_MARKERS, block_flags, build_prefill_prompt
from src.common.sampling import split_sampling
from src.common.split import first_sentence

PILES = ("kept", "replaced")


def pile_prompt(reader: str, trace, i: int, pile: str) -> str:
    return build_prefill_prompt(reader, trace.question, trace.chunks[: i + 1 if pile == "kept" else i])


def trace_key(trace) -> str:
    return f"{trace.split}/{trace.problem_id}"


def existing(rows_path, trace, i: int, pile: str) -> list[dict]:
    """Rows on disk for one pile. Lines are pre-filtered on the literal text
    json.dumps wrote for the row's identity (field order fixed in generate), so
    a multi-GB rows file costs a scan, not a parse per call."""
    needle = f'"trace": "{trace_key(trace)}", "chunk_idx": {i}, "pile": "{pile}"'
    p = Path(rows_path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            if needle in line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:      # a truncated last line, as read_rows skips
                    continue
    return out


def generate(client, cfg, reader, trace, i, pile, n, samp, rows_path, run_id, phase, filt, topup, lock=None) -> list[dict]:
    """n rollouts for one pile, written as they land. Seed salted by what is
    already on disk, so a resumed or topped-up pile never reuses a seed."""
    done = len(existing(rows_path, trace, i, pile))
    prompt = pile_prompt(reader, trace, i, pile)
    seed = rng(cfg["seed"], phase, reader, trace_key(trace), i, pile, done, run_id).getrandbits(31)   # run_id in the salt since 2026-09-10 (batch 2 must not reuse batch 1's seeds)
    T, top_p, cap, extra = split_sampling(samp)
    out = []
    for k, c in enumerate(client.complete(prompt, n, cap, T, top_p, seed=seed, **extra)):
        answer = extract_answer(c["text"])
        flags = block_flags(reader, c["text"], c["finish_reason"])
        row = {"run_id": run_id, "phase": phase, "reader": reader, "trace": trace_key(trace), "chunk_idx": i,
               "pile": pile, "sample_idx": done + k, "seed": seed, "topup": topup, "prompt": prompt,
               "completion": c["text"], "finish_reason": c["finish_reason"], "tokens": c["tokens"], **flags,
               "answer": answer, "correct": is_correct(answer, trace.gt_answer),
               "ts": dt.datetime.now(dt.UTC).isoformat()}
        if pile == "replaced":
            # Only the reasoning block can replace A's sentence: cut at the reader's closer first, as the
            # dataset's splitter cuts A's rollouts at </think> (split.py:35-36) - a rollout that closes at
            # once has no replacement and is not usable. For gpt-oss the closer is <|end|>, which the
            # splitter does not know.
            rep = first_sentence(c["text"].split(REASONING_MARKERS[reader][1])[0])
            sim = filt.similarity(rep, trace.chunks[i]) if rep else None
            row.update({"first_sentence": rep, "similarity": sim,
                        "dissimilar": bool(rep) and filt.is_dissimilar(sim)})
        if lock:
            with lock:
                append_row(rows_path, row)
        else:
            append_row(rows_path, row)
        out.append(row)
    return out


def usable(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r["reopened"] and (r["pile"] == "kept" or r.get("dissimilar"))]


def ensure_pile(client, cfg, reader, trace, i, pile, samp, rows_path, run_id, phase, filt, lock=None) -> list[dict]:
    """Bring one pile up to the design's counts: base size, then the top-up rules."""
    imp = cfg["importance"]
    base = imp["kept_samples"] if pile == "kept" else imp["replaced_samples"]
    rows = existing(rows_path, trace, i, pile)
    want = pile_prompt(reader, trace, i, pile)          # a resumed run must not mix prompts
    assert all(r["prompt"] == want for r in rows), f"{run_id}: rows on disk for {trace_key(trace)} chunk {i} {pile} were generated with a different prompt; use a new run_id"
    def fill(topup, target):
        """Bring the rows carrying this top-up flag up to `target` - so a top-up
        interrupted by a kill is completed on resume, not left short."""
        have = sum(r["topup"] == topup for r in rows)
        if have < target:
            rows.extend(generate(client, cfg, reader, trace, i, pile, target - have, samp, rows_path, run_id, phase, filt, topup, lock))

    fill(0, base)
    started = lambda topup: any(r["topup"] == topup for r in rows)     # a top-up that began is always completed
    if pile == "replaced" and (started(1) or len(usable(rows)) < imp["replaced_min_survivors"]):
        fill(1, imp["replaced_topup_samples"])
    if started(2) or p_correct(usable(rows))["n_answered"] < imp["min_answered_per_pile"]:
        fill(2, imp["answered_topup_samples"])
    return rows


def complete(rows: list[dict], pile: str, imp: dict) -> bool:
    """True iff ensure_pile would request nothing more: base rows present, and
    each top-up either not needed or fully written. The analysis treats any
    other pile as unfinished."""
    n_topup = lambda k: sum(r["topup"] == k for r in rows)
    if n_topup(0) < (imp["kept_samples"] if pile == "kept" else imp["replaced_samples"]):
        return False
    if pile == "replaced" and (n_topup(1) or len(usable(rows)) < imp["replaced_min_survivors"]) and n_topup(1) < imp["replaced_topup_samples"]:
        return False                                  # top-up 1 needed or started, and not fully written
    if (n_topup(2) or p_correct(usable(rows))["n_answered"] < imp["min_answered_per_pile"]) and n_topup(2) < imp["answered_topup_samples"]:
        return False
    return True


def pile_stats(rows: list[dict]) -> dict:
    u = usable(rows)
    pc = p_correct(u)
    return {"n": len(rows), "n_reopened": sum(bool(r["reopened"]) for r in rows),
            "n_dissimilar": sum(bool(r.get("dissimilar")) for r in rows) if rows and rows[0]["pile"] == "replaced" else None,
            "n_usable": len(u), "n_answered": pc["n_answered"], "n_no_answer": pc["n_no_answer"],
            "n_capped": sum(bool(r["capped"]) for r in rows), "p_correct": pc["p_correct"]}


def importance(kept_rows: list[dict], replaced_rows: list[dict]) -> dict:
    """Signed, the source's sign order: P(correct | replaced) − P(correct | kept)."""
    k, r = pile_stats(kept_rows), pile_stats(replaced_rows)
    imp = None if k["p_correct"] is None or r["p_correct"] is None else r["p_correct"] - k["p_correct"]
    return {"kept": k, "replaced": r, "importance": imp}
