#!/usr/bin/env python
"""Verbatim examples for reading. do not summarise raw rollouts
on the author's behalf - show N seeded random rows and let the author read them.

    python scripts/91_sample_examples.py --seed 20260908 [--reader gpt-oss-20b]
        [--n 3] [--out doc/examples.md] [--config configs/phase1.json]

Four blocks, every one of them raw text off disk, cut only at an explicit
[... truncated N characters ...] marker and never paraphrased:
  1 anchor/partner pairs (data/p1_pairs.csv, sentences from data/p1_importance.csv):
    A's importance at both sentences, the reader's, d, and both sentences in full.
  2 kept-pile continuations (runs/p1_<reader>/rows.jsonl): A's last prefilled
    sentence, how the reader carried on, and whether it answered correctly.
  3 replacement sentences the semantic filter ACCEPTED and REJECTED, each with its
    cosine to A's sentence and the threshold (configs/phase1.json similarity_max_cosine).
  4 rollouts where the reader abandoned the prefilled reasoning block, if it has any.

Each block draws from rng(seed, "91", <block>, reader), so a block can be added or
re-run without moving the others, and every example carries the identity of its
row (run id, trace, chunk index, pile, sample_idx) plus the grep that finds it again.
"""

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data import ROLLOUTS_ROOT, Trace
from src.common.io import REPO_ROOT, load_config, read_rows, rng, run_dir
from src.common.prompts import REASONING_MARKERS, _HARMONY_REOPEN   # the matcher that set each row's `reopened` flag

DATA = REPO_ROOT / "data"
CONTINUATION_CHARS = 1200                 # of a continuation, before the truncation marker
REOPEN_BEFORE, REOPEN_AFTER = 400, 800    # window around the point where the block was abandoned
DIRECTION = {-1: "A's own sentence helped A (importance_A < 0): a reader that rewrites it should do WORSE",
             +1: "A's own sentence hurt A (importance_A > 0): a reader that rewrites it should do BETTER"}


def csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def excerpt(text, lo, hi):
    """text[lo:hi], with an explicit marker for whatever was cut off each end."""
    lo, hi = max(0, lo), min(len(text), hi)
    head = f"[... {lo} earlier characters omitted ...]\n" if lo else ""
    tail = f"\n[... truncated, {len(text) - hi} more characters ...]" if hi < len(text) else ""
    return head + text[lo:hi] + tail


def fence(text):
    """A fenced block whose fence is longer than any run of backticks inside, so a
    completion that itself contains ``` still pastes into a doc intact."""
    ticks = "`" * max(3, max((len(m) for m in re.findall("`+", text)), default=0) + 1)
    return f"{ticks}\n{text}\n{ticks}"


def needle(r):
    """The grep that finds this row again: the literal text json.dumps wrote for its
    identity (field order fixed in piles.generate), as piles.existing matches on."""
    return f'"trace": "{r["trace"]}", "chunk_idx": {r["chunk_idx"]}, "pile": "{r["pile"]}", "sample_idx": {r["sample_idx"]},'


def role_label(t):
    """How data/p1_importance.csv names the sentence: 'anchor 2' / 'partner of anchor 2'."""
    return f"anchor {t['anchor_rank']}" if t["role"] == "anchor" else f"partner of anchor {t['anchor_rank']}"


def outcome(r):
    """What the rollout's answer was worth. P(correct) is answered-only (configs/
    phase1.json p_correct_denominator), so a rollout that never boxed an answer is
    dropped from both piles, not counted wrong."""
    if not r["answer"]:
        return "NO ANSWER (dropped; P(correct) is answered-only)"
    return "CORRECT" if r["correct"] else "WRONG"


def reopen_at(reader, text):
    """Index where the reader abandoned the prefilled block, or None. Same markers as
    src/common/prompts.block_flags, so the window shown is the one that set the flag."""
    if reader == "gpt-oss-20b":
        m = _HARMONY_REOPEN.search(text)
        return m.start() if m else None
    i = text.find(REASONING_MARKERS[reader][0])
    return i if i >= 0 else None


def pairs_block(reader, draw, n):
    """Block 1: whole anchor/partner pairs, so the author can judge whether the anchor
    reads like a turning point and the partner like an inert neighbour."""
    pairs = [p for p in csv_rows(DATA / "p1_pairs.csv") if p["reader"] == reader and p["d"]]
    sent = {(t["split"], t["problem_id"], t["anchor_rank"], t["role"]): t
            for t in csv_rows(DATA / "p1_importance.csv") if t["reader"] == reader}
    md = [f"## 1. Anchor / partner pairs ({min(n, len(pairs))} drawn of {len(pairs)} pairs with a defined d)", "",
          "Every importance is in ANSWER ACCURACY: P(correct | the sentence was rewritten) - P(correct | A's",
          "sentence kept), so -0.10 means ten accuracy points lost. `d` = sign_A x (importance_B at the anchor -",
          "importance_B at the partner); d > 0 means the reader swung the way A's data predicts, more at the",
          "anchor than at the control.", ""]
    ids = []
    for p in draw.sample(pairs, min(n, len(pairs))):
        a = sent[(p["split"], p["problem_id"], p["anchor_rank"], "anchor")]
        q = sent[(p["split"], p["problem_id"], p["anchor_rank"], "partner")]
        assert (a["chunk_idx"], q["chunk_idx"]) == (p["anchor_chunk"], p["partner_chunk"]), f"{p['problem_id']}: p1_pairs.csv and p1_importance.csv disagree on which sentences the pair uses"
        d = int(p["direction"]) * (float(p["importance_B_anchor"]) - float(p["importance_B_partner"]))
        assert abs(d - float(p["d"])) < 1e-9, f"{p['problem_id']}: recomputed d {d} != data/p1_pairs.csv d {p['d']}"
        ids.append(f"{'pair':9}{p['split']}/{p['problem_id']} anchor {p['anchor_rank']} (chunks {p['anchor_chunk']}/{p['partner_chunk']})")
        md += [f"### 1.{len(ids)} {p['split']}/{p['problem_id']}, anchor {p['anchor_rank']}", "",
               f"`data/p1_pairs.csv`: reader {reader}, anchor chunk {p['anchor_chunk']}, partner chunk {p['partner_chunk']}, "
               f"direction {p['direction']} - {DIRECTION[int(p['direction'])]}", "",
               "| sentence | chunk | A's importance | this reader's importance |", "|---|---|---|---|",
               f"| anchor ({p['anchor_tags']}) | {p['anchor_chunk']} | {float(p['importance_A_anchor']):+.4f} | {float(p['importance_B_anchor']):+.4f} |",
               f"| partner | {p['partner_chunk']} | {float(p['importance_A_partner']):+.4f} | {float(p['importance_B_partner']):+.4f} |", "",
               f"d = {p['direction']} x ({float(p['importance_B_anchor']):+.4f} - {float(p['importance_B_partner']):+.4f}) = "
               f"**{float(p['d']):+.4f}** accuracy"
               + ("  \n(`short` = True: a pile answered fewer times than min_answered_per_pile after the top-ups)" if p["short"] == "True" else ""), "",
               f"A's sentence {p['anchor_chunk']}, the ANCHOR:", fence(a["text"]), "",
               f"A's sentence {p['partner_chunk']}, the PARTNER:", fence(q["text"]), ""]
    return md, ids


def kept_block(run_id, rows, traces, roles, draw, n):
    """Block 2: the reader carrying on from A's prefix, kept pile."""
    pool = [r for r in rows if r["pile"] == "kept" and not r["reopened"]]
    md = [f"## 2. Kept-pile continuations ({min(n, len(pool))} drawn of {len(pool)} kept rollouts that did not re-open the block)", "",
          "The kept pile prefills A's sentences 0..i and lets the reader continue (src/phase1/piles.py:4-5), so the",
          "last sentence of the prefix is A's sentence i - printed first for context, then the reader's completion.", ""]
    ids = []
    for r in draw.sample(pool, min(n, len(pool))):
        t, i = traces[r["trace"]], r["chunk_idx"]
        ids.append(f"{'kept':9}{run_id} {r['trace']} chunk {i} sample_idx {r['sample_idx']}")
        md += [f"### 2.{len(ids)} {r['trace']}, chunk {i} ({roles[(r['trace'], i)]}), kept, sample_idx {r['sample_idx']}", "",
               f"`grep '{needle(r)}' runs/{run_id}/rows.jsonl`  \n"
               f"answer {r['answer']!r} vs ground truth {t.gt_answer!r} -> **{outcome(r)}**; "
               f"{r['tokens']} tokens; finish_reason {r['finish_reason']}", "",
               f"A's sentence {i}, the last of the prefilled prefix:", fence(t.chunks[i]), "",
               "The reader continues:", fence(excerpt(r["completion"], 0, CONTINUATION_CHARS)), ""]
    return md, ids


def filter_block(run_id, rows, traces, roles, draw, n, threshold):
    """Block 3: what the semantic filter accepts and rejects. A replaced rollout counts
    only if the reader's first sentence is DIFFERENT enough from A's (cosine < threshold)."""
    pool = [r for r in rows if r["pile"] == "replaced" and not r["reopened"] and r["first_sentence"]]
    accepted = [r for r in pool if r["dissimilar"]]
    rejected = [r for r in pool if not r["dissimilar"]]
    md = [f"## 3. What the semantic filter did ({min(n, len(accepted))} accepted, {min(n, len(rejected))} rejected)", "",
          f"A replaced rollout counts iff cosine(the reader's first sentence, A's sentence) < {threshold} "
          f"(all-MiniLM-L6-v2, src/common/filters.py). Of {len(pool)} replaced rollouts that did not re-open the",
          f"block, {len(accepted)} were accepted and {len(rejected)} rejected as too similar to A's sentence.", ""]
    ids = []
    for verdict, pick in (("ACCEPTED", accepted), ("REJECTED", rejected)):
        for r in sorted(draw.sample(pick, min(n, len(pick))), key=lambda r: r["similarity"]):
            t, i = traces[r["trace"]], r["chunk_idx"]
            ids.append(f"{verdict.lower():9}{run_id} {r['trace']} chunk {i} sample_idx {r['sample_idx']} cosine {r['similarity']:.3f}")
            md += [f"### 3.{len(ids)} {verdict} - cosine {r['similarity']:.3f} {'<' if r['dissimilar'] else '>='} {threshold}", "",
                   f"{r['trace']}, chunk {i} ({roles[(r['trace'], i)]}), replaced, sample_idx {r['sample_idx']}  \n"
                   f"`grep '{needle(r)}' runs/{run_id}/rows.jsonl`", "",
                   f"A's sentence {i}:", fence(t.chunks[i]), "",
                   "The reader's replacement (first sentence of its continuation):", fence(r["first_sentence"]), ""]
    return md, ids


def reopen_block(reader, run_id, rows, roles, draw, n):
    """Block 4: rollouts that abandoned the prefilled reasoning block - a known failure
    mode, shown rather than described. Such rollouts are dropped."""
    pool = [r for r in rows if r["reopened"]]
    if not pool:
        return ["## 4. Re-opened reasoning blocks (none)", "",
                f"No rollout of {reader} in runs/{run_id} abandoned the prefilled reasoning block "
                f"(0 of {len(rows)}), so there is nothing to show.", ""], []
    md = [f"## 4. Re-opened reasoning blocks ({min(n, len(pool))} drawn of the {len(pool)} that re-opened, of {len(rows)} rollouts)", "",
          "The reader closed the reasoning block it was handed and started its own (src/common/prompts.block_flags);",
          "it is then no longer continuing A's reasoning, so the rollout is dropped from P(correct). The window",
          "below is centred on the point where it happened.", ""]
    ids = []
    for r in draw.sample(pool, min(n, len(pool))):
        i, at = r["chunk_idx"], reopen_at(reader, r["completion"])
        assert at is not None, f"row flagged reopened but no marker found: {needle(r)}"
        ids.append(f"{'reopened':9}{run_id} {r['trace']} chunk {i} {r['pile']} sample_idx {r['sample_idx']} channel {r['reopen_channel']}")
        md += [f"### 4.{len(ids)} {r['trace']}, chunk {i} ({roles[(r['trace'], i)]}), {r['pile']}, sample_idx {r['sample_idx']}", "",
               f"`grep '{needle(r)}' runs/{run_id}/rows.jsonl`  \n"
               f"re-opened in channel {r['reopen_channel']!r} at character {at} of {len(r['completion'])}; "
               f"answer {r['answer']!r} -> {outcome(r)}, but the rollout is dropped either way", "",
               "The completion around that point:",
               fence(excerpt(r["completion"], at - REOPEN_BEFORE, at + REOPEN_AFTER)), ""]
    return md, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="gpt-oss-20b")
    ap.add_argument("--n", type=int, default=3, help="examples per block (block 4 takes min(n, 2))")
    ap.add_argument("--seed", type=int, default=None, help="default: the config seed")
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--out", default="doc/examples.md")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    threshold = cfg["importance"]["similarity_max_cosine"]
    run_id = f"p1_{args.reader}"
    rows_path = run_dir(run_id, create=False) / "rows.jsonl"
    rows = list(read_rows(rows_path))
    assert rows, f"no rows under {rows_path}"
    traces = {k: Trace(ROLLOUTS_ROOT / k.split("/")[0] / k.split("/")[1], k.split("/")[0])
              for k in {r["trace"] for r in rows}}
    roles = {(f"{t['split']}/{t['problem_id']}", int(t["chunk_idx"])): role_label(t)
             for t in csv_rows(DATA / "p1_importance.csv") if t["reader"] == args.reader}

    draw = lambda block: rng(seed, "91", block, args.reader)
    blocks = [pairs_block(args.reader, draw("pairs"), args.n),
              kept_block(run_id, rows, traces, roles, draw("kept"), args.n),
              filter_block(run_id, rows, traces, roles, draw("filter"), args.n, threshold),
              reopen_block(args.reader, run_id, rows, roles, draw("reopen"), min(args.n, 2))]

    md = [f"# Verbatim examples - reader {args.reader}", "",
          "Raw text off disk, never paraphrased; a cut is marked `[... truncated N characters ...]`.", "",
          f"- selection: `python scripts/91_sample_examples.py --seed {seed} --reader {args.reader} --n {args.n}`, "
          f'RNG `src/common/io.rng({seed}, "91", <block>, "{args.reader}")`',
          f"- rollouts: `runs/{run_id}/rows.jsonl` ({len(rows)} rows, {len(traces)} traces)",
          "- pairs and sentences: `data/p1_pairs.csv`, `data/p1_importance.csv`",
          f"- semantic filter: keep iff cosine < {threshold} (`{args.config}` importance.similarity_max_cosine)",
          f"- generated {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC", "",
          "Importances are in answer accuracy, a probability in [0, 1]: +0.05 is five accuracy points.", ""]
    ids = []
    for lines, got in blocks:
        md += lines
        ids += got

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out} ({len(md)} lines) - seed {seed}, reader {args.reader}, {len(ids)} examples")
    for i in ids:
        print("  " + i)


if __name__ == "__main__":
    main()
