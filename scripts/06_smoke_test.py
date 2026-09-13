#!/usr/bin/env python
"""Smoke test: one reader on one of ITS selected traces,
3 sentences (the trace's anchors) x 8 rollouts per pile, through the same pile
code as Phase 1 (src/phase1/piles: prompts, sampling, splitter, semantic
filter, answer extraction, re-open flags) - no top-ups.

    python scripts/06_smoke_test.py --reader gpt-oss-20b [--config configs/phase1.json]
        [--run_id smoke_<reader>] [--n 8] [--trace split/problem_id] [--workers 6] [--ack]

Writes runs/<run_id>/rows.jsonl (one row per rollout, resumable), and
runs/<run_id>/continuations.md with EVERY completion in full under a header of
its flags - the author reads that file, the script does not summarise it.
Prints per pile: n, re-opened, capped, answered, correct, dissimilar; then 10
replacement sentences (seeded: up to 5 kept, 5 discarded) with their cosine
to A's sentence. The trace is a seeded pick from the reader's list in
data/phase1_traces.json, recorded in data/smoke_trace_<reader>.json.
CHECKPOINT: read continuations.md (does the reader continue A's block?
re-open rate? are the filter's keep/discard decisions sensible?). Rerun with
--ack to record.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.client import VLLMClient
from src.common.data import ROLLOUTS_ROOT, Trace
from src.common.filters import SemanticFilter
from src.common.io import REPO_ROOT, load_config, rng, run_dir, write_run_config
from src.common.sampling import reader_sampling
from src.phase1.piles import PILES, existing, generate, pile_prompt, pile_stats, usable


def pick_trace(cfg, reader) -> tuple[str, list[dict]]:
    sel = json.load(open(REPO_ROOT / "data/phase1_traces.json"))["by_reader"][reader]["traces"]
    ids = sorted(f"{t['split']}/{t['problem_id']}" for t in sel)
    choice = rng(cfg["seed"], "smoke_trace", reader).choice(ids)
    (REPO_ROOT / f"data/smoke_trace_{reader}.json").write_text(json.dumps(
        {"seed": cfg["seed"], "salt": ["smoke_trace", reader], "candidates": ids, "trace": choice}, indent=2))
    return choice, sel


def header(r) -> str:
    flags = ["REOPENED" if r["reopened"] else "continued", "closed" if r["closed"] else "not closed",
             "CAPPED" if r["capped"] else r["finish_reason"], f"{r['tokens']} tokens",
             f"answer={r['answer']!r}", "correct" if r["correct"] else "wrong"]
    if r["pile"] == "replaced":
        sim = "None" if r["similarity"] is None else f"{r['similarity']:.2f}"
        flags.append(f"{'KEPT' if r['dissimilar'] else 'DISCARDED'} (cosine {sim})")
    return " | ".join(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--n", type=int, default=8, help="rollouts per pile (8 is enough for a smoke test)")
    ap.add_argument("--trace", default=None, help="split/problem_id; default: seeded pick from the reader's list")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--ack", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_id = args.run_id or f"smoke_{args.reader}"
    trace_id, sel = pick_trace(cfg, args.reader) if args.trace is None else (args.trace, json.load(open(REPO_ROOT / "data/phase1_traces.json"))["by_reader"][args.reader]["traces"])
    split, pid = trace_id.split("/")
    trace = Trace(ROLLOUTS_ROOT / split / pid, split)
    pairs = next((t for t in sel if t["split"] == split and t["problem_id"] == pid), None)
    assert pairs is not None, f"{trace_id} is not in {args.reader}'s list in data/phase1_traces.json"
    pairs = pairs["pairs"]
    samp = reader_sampling(cfg, args.reader)
    rows_path = run_dir(run_id) / "rows.jsonl"
    client = VLLMClient(args.reader, timeout=args.timeout)
    client.assert_serving()
    write_run_config(run_id, cfg, extra={"script": "06_smoke_test", "args": vars(args), "trace": trace_id,
                                         "anchors": [p["anchor_chunk"] for p in pairs], "sampling_used": samp})
    print(f"{args.reader} on {trace_id} ({trace.n_chunks} sentences, A {'correct' if trace.base_is_correct else 'wrong'}) | "
          f"anchors {[p['anchor_chunk'] for p in pairs]} | {args.n}/pile | {samp}")

    imp = cfg["importance"]
    filt = SemanticFilter(imp["similarity_model"], imp["similarity_device"], imp["similarity_max_cosine"])
    lock = threading.Lock()

    def job(i, pile):
        rows = existing(rows_path, trace, i, pile)
        assert all(r["prompt"] == pile_prompt(args.reader, trace, i, pile) for r in rows), f"rows on disk for chunk {i} {pile} were made with another prompt; use a new run_id"
        if len(rows) < args.n:
            rows += generate(client, cfg, args.reader, trace, i, pile, args.n - len(rows), samp, rows_path, run_id, "smoke", filt, 0, lock)
        return rows

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, p["anchor_chunk"], pile): (p["anchor_chunk"], pile) for p in pairs for pile in PILES}
        piles = {futs[f]: f.result() for f in futs}
    print(f"{sum(len(v) for v in piles.values())} rows in {time.time() - t0:.0f}s -> {rows_path.relative_to(REPO_ROOT)}")

    md = [f"# Smoke test: {args.reader} on {trace_id}\n"]
    print(f"\n{'anchor':>6} {'pile':>8} {'n':>3} {'reopened':>8} {'capped':>6} {'dissimilar':>10} {'usable':>6} {'answered':>8} {'correct':>7}   (usable = not re-opened and, if replaced, dissimilar; answered/correct count usable rows)")
    for p in pairs:
        i = p["anchor_chunk"]
        md.append(f"\n## Sentence {i} (0-indexed chunk; {'+' if p['anchor_direction'] > 0 else '-'}, A's cf-importance {p['anchor_cf_importance']:+.3f})\n\nA's sentence: {trace.chunks[i]!r}\n")
        for pile in PILES:
            rows = sorted(piles[(i, pile)], key=lambda r: r["sample_idx"])
            st = pile_stats(rows)
            print(f"{i:>6} {pile:>8} {st['n']:>3} {st['n_reopened']:>8} {st['n_capped']:>6} {'-' if st['n_dissimilar'] is None else st['n_dissimilar']:>10} "
                  f"{st['n_usable']:>6} {st['n_answered']:>8} {sum(bool(r['correct']) for r in usable(rows) if r['answer']):>7}")
            md.append(f"\n### {pile} pile (chunks[:{i + 1 if pile == 'kept' else i}] prefilled)\n")
            for r in rows:
                md.append(f"\n#### sample {r['sample_idx']}: {header(r)}\n\n~~~~\n{r['completion']}\n~~~~\n")   # tilde fence: completions may contain ```
    (run_dir(run_id) / "continuations.md").write_text("".join(md))

    rep = [r for (i, pile), rs in piles.items() if pile == "replaced" for r in rs if r["first_sentence"] and not r["reopened"]]
    draw = rng(cfg["seed"], "smoke_show", args.reader)
    show = [r for keep in (True, False) for r in draw.sample([r for r in rep if bool(r["dissimilar"]) == keep], min(5, sum(bool(r["dissimilar"]) == keep for r in rep)))]
    print(f"\n{len(show)} replacement sentences (seeded; kept = cosine < {imp['similarity_max_cosine']} to A's sentence):")
    for r in sorted(show, key=lambda r: r["similarity"]):
        print(f"  [{'KEEP' if r['dissimilar'] else 'DROP'} {r['similarity']:.2f}] sentence {r['chunk_idx']}\n"
              f"      A: {trace.chunks[r['chunk_idx']][:160]!r}\n      B: {r['first_sentence'][:160]!r}")
    print(f"\nfull continuations: {(run_dir(run_id) / 'continuations.md').relative_to(REPO_ROOT)}")
    if not args.ack:
        print("\nCHECKPOINT (the calibration run smoke): read every continuation in continuations.md; does the reader continue A's block, how often does it "
              "re-open, are the keep/discard decisions sensible? Rerun with --ack to record.")
        sys.exit(2)
    (run_dir(run_id) / "ack.txt").write_text(time.strftime("%Y-%m-%d %H:%M UTC\n", time.gmtime()))
    print("acknowledged")


if __name__ == "__main__":
    main()
