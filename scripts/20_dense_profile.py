#!/usr/bin/env python
"""Dense importance profile of a reader on A's trace, the original study's way
(accuracy and the reader's own importance at EVERY sentence).

    python scripts/20_dense_profile.py --reader gpt-oss-20b [--split correct_base_solution]
        [--n 100] [--config configs/phase1.json]
        [--run_id dense_<reader>] [--workers 10] [--timeout 7200]

At every sentence i of each trace, n rollouts from the prefix chunks[:i] - the
"replaced" pile of src/phase1/piles (the reader writes its own sentence i and
continues; semantic filter applied) - which is what generate_rollouts.py did
for A, 100 per sentence. From those rollouts, re-opened ones excluded (the pre-registration):
accuracy(i) = P(correct | answered); counterfactual importance(i) =
P(correct | dissimilar at i, answered) − accuracy(i+1) (analyze_rollouts.py:773);
resampling importance(i) = accuracy(i+1) − accuracy(i). Rows go to
runs/<run_id>/rows.jsonl as they land (resumable per sentence, prompt-checked);
runs/<run_id>/profile.csv holds one row per sentence with the reader's
quantities beside A's released ones and the sentence's role in the anchor
table; it is rewritten after every trace (A's cf importance is None where A never
measured it, data.py is_measured). Traces default to the reader's
selected traces of the split (data/phase1_traces.json); one trace at a time, in
order, so a finished trace is usable early. Figure: 90_make_figures.py --only fig7b.
"""

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.answers import p_correct
from src.common.client import VLLMClient
from src.common.data import ROLLOUTS_ROOT, Trace
from src.common.filters import SemanticFilter
from src.common.io import REPO_ROOT, load_config, read_rows, run_dir, write_run_config
from src.common.prompts import pin_render_date, render_date
from src.common.sampling import reader_sampling
from src.phase1.piles import existing, generate, pile_prompt, trace_key


def sentence_pile(client, cfg, reader, trace, i, n, samp, rows_path, run_id, filt, lock):
    rows = existing(rows_path, trace, i, "replaced")
    want = pile_prompt(reader, trace, i, "replaced")
    assert all(r["prompt"] == want for r in rows), f"{run_id}: rows on disk for {trace_key(trace)} chunk {i} were made with another prompt; use a new run_id"
    if len(rows) < n:
        rows += generate(client, cfg, reader, trace, i, "replaced", n - len(rows), samp, rows_path, run_id, "dense", filt, 0, lock)
    return rows


def profile(reader, trace, rows_by_i, roles) -> list[dict]:
    """One row per sentence: the reader's accuracy and importances from its rollouts, A's released values beside them."""
    out = []
    stats = {}
    for i, rows in rows_by_i.items():
        kept = [r for r in rows if not r["reopened"]]
        dis = [r for r in kept if r.get("dissimilar")]
        stats[i] = {"n": len(rows), "n_reopened": len(rows) - len(kept), "n_capped": sum(bool(r["capped"]) for r in rows),
                    **{f"{k}": v for k, v in p_correct(kept).items()}, "n_dissimilar": len(dis),
                    "n_dissimilar_answered": p_correct(dis)["n_answered"], "p_correct_dissimilar": p_correct(dis)["p_correct"]}
    for i in sorted(stats):
        s, nxt = stats[i], stats.get(i + 1)
        acc, acc_next = s["p_correct"], None if nxt is None else nxt["p_correct"]
        lab = trace.chunks_labeled[i]
        out.append({"reader": reader, "split": trace.split, "problem_id": trace.problem_id, "chunk_idx": i,
                    "n": s["n"], "n_reopened": s["n_reopened"], "n_capped": s["n_capped"], "n_answered": s["n_answered"],
                    "accuracy": acc, "n_dissimilar": s["n_dissimilar"], "n_dissimilar_answered": s["n_dissimilar_answered"],
                    "p_correct_dissimilar": s["p_correct_dissimilar"],
                    "cf_importance": None if s["p_correct_dissimilar"] is None or acc_next is None else s["p_correct_dissimilar"] - acc_next,
                    "resampling_importance": None if acc is None or acc_next is None else acc_next - acc,
                    "dtf": (s["n_dissimilar_answered"] / s["n_answered"]) if s["n_answered"] else None,
                    "released_accuracy": lab.get("accuracy"), "released_cf_importance": trace.importance_a(i),   # None where A never measured it
                    "released_dtf": lab.get("different_trajectories_fraction"), "released_testable": trace.testable(i),
                    "tags": "|".join(lab.get("function_tags", [])),
                    "role": roles.get(i, ""), "text": trace.chunks[i]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--split", default="correct_base_solution")
    ap.add_argument("--n", type=int, default=100, help="rollouts per sentence (the original study: 100)")
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_id = args.run_id or f"dense_{args.reader}"
    sel = [t for t in json.load(open(REPO_ROOT / "data/phase1_traces.json"))["by_reader"][args.reader]["traces"] if t["split"] == args.split]
    assert sel, f"{args.reader} has no {args.split} traces in data/phase1_traces.json"
    samp = reader_sampling(cfg, args.reader)
    rows_path = run_dir(run_id) / "rows.jsonl"
    pin_render_date(next((d for r in read_rows(rows_path) if (d := render_date(r["prompt"]))), None))   # resume with the run's own prompts
    client = VLLMClient(args.reader, timeout=args.timeout)
    client.assert_serving()
    write_run_config(run_id, cfg, extra={"script": "20_dense_profile", "args": vars(args), "sampling_used": samp,
                                         "traces": [f"{t['split']}/{t['problem_id']}" for t in sel]})
    stored = json.load(open(run_dir(run_id) / "config.json"))
    assert stored["importance"] == cfg["importance"] and stored["_run"]["sampling_used"] == samp and stored["_run"]["args"]["n"] == args.n \
        and stored["_run"]["traces"] == [f"{t['split']}/{t['problem_id']}" for t in sel], \
        f"runs/{run_id}/config.json differs from this call (importance block, sampling, n or trace list); use a new run_id"
    foreign = [r for r in read_rows(rows_path) if r["run_id"] != run_id or r["reader"] != args.reader or r["phase"] != "dense"]
    assert not foreign, f"runs/{run_id}/rows.jsonl holds {len(foreign)} rows from another run/reader/phase"
    total = sum(t["n_chunks"] for t in sel)
    print(f"{args.reader} on {len(sel)} {args.split} trace(s), {total} sentences x {args.n} rollouts | {samp} -> {rows_path.relative_to(REPO_ROOT)}")

    imp = cfg["importance"]
    filt = SemanticFilter(imp["similarity_model"], imp["similarity_device"], imp["similarity_max_cosine"])
    lock = threading.Lock()
    t0, failed, prof = time.time(), [], []
    for t in sel:
        trace = Trace(ROLLOUTS_ROOT / t["split"] / t["problem_id"], t["split"])
        roles = {}
        for p in t["pairs"]:
            roles[p["anchor_chunk"]] = f"anchor {p['anchor_rank']}"
            roles[p["partner_chunk"]] = f"partner of {p['anchor_chunk']}"
        rows_by_i = {}
        for r in read_rows(rows_path):                 # one pass per trace; finished sentences skip the pool (and its re-reads)
            if r["trace"] == trace_key(trace):
                rows_by_i.setdefault(r["chunk_idx"], []).append(r)
        rows_by_i = {i: rs for i, rs in rows_by_i.items() if len(rs) >= args.n}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(sentence_pile, client, cfg, args.reader, trace, i, args.n, samp, rows_path, run_id, filt, lock): i
                    for i in range(trace.n_chunks) if i not in rows_by_i}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    rows_by_i[i] = f.result()
                except Exception as e:
                    failed.append(f"{t['split']}/{t['problem_id']} chunk {i}")
                    print(f"[{time.time() - t0:>6.0f}s] FAILED {failed[-1]}: {e!r}", flush=True)
                    continue
                pc = p_correct([r for r in rows_by_i[i] if not r["reopened"]])
                print(f"[{time.time() - t0:>6.0f}s] {t['problem_id']} sentence {i:>3}/{trace.n_chunks}: n {len(rows_by_i[i])} answered {pc['n_answered']} "
                      f"acc {'-' if pc['p_correct'] is None else f'{pc['p_correct']:.2f}'} dissimilar {sum(bool(r.get('dissimilar')) for r in rows_by_i[i] if not r['reopened'])}"
                      + (f" reopened {sum(r['reopened'] for r in rows_by_i[i])}" if any(r['reopened'] for r in rows_by_i[i]) else ""), flush=True)
        done = profile(args.reader, trace, rows_by_i, roles)
        prof += done
        if not done:
            continue
        csv_path = run_dir(run_id) / "profile.csv"      # merge: this trace's rows replace their old version, other traces stay
        others = [r for r in csv.DictReader(open(csv_path))] if csv_path.exists() else []
        others = [r for r in others if (r["split"], r["problem_id"]) != (t["split"], t["problem_id"])]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(done[0]))
            w.writeheader()
            w.writerows(others + done)
        print(f"\n{t['split']}/{t['problem_id']}: {len(done)} sentences profiled -> runs/{run_id}/profile.csv "
              f"({sum(1 for r in done if r['cf_importance'] is None)} without a defined importance)\n", flush=True)
    if failed:
        print(f"{len(failed)} sentence piles failed; rerun the same command to resume them:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print(f"done: {len(prof)} sentences in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
