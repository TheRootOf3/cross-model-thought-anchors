#!/usr/bin/env python
"""Saturation screen: which candidate readers are neither saturated nor hopeless.

    python scripts/03_screen_readers.py --reader qwen3.5-9b [--config configs/phase1.json]
        [--run_id screen_qwen3.5-9b] [--workers 4] [--timeout 3600]

One reader answers each of the 20 dataset problems from the problem alone (no
prefix: its own chat template, thinking enabled, its own reasoning opener,
nothing prefilled), `saturation_screen.n_samples` times, with its recommended
sampling (configs/models.json) at the pre-registered design cap. Every rollout is one row in
runs/<run_id>/rows.jsonl, written as it lands; rerunning resumes per problem.
Then per problem: P(correct | answered) with n_answered beside it (the
thought-anchors denominator, src/common/answers.p_correct), cap hits, re-opens,
and the band flag band_low <= P <= band_high. The summary rows for this reader
replace any earlier ones in data/screen_summary.csv. Prints the in-band list
and whether it reaches min_surviving_problems. No CHECKPOINT here; the
human reads data/screen_summary.csv and decides B_near per the pre-registered design
"""

import argparse
import datetime as dt
import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.answers import extract_answer, is_correct, p_correct
from src.common.client import VLLMClient
from src.common.data import load_traces
from src.common.io import REPO_ROOT, append_row, load_config, read_rows, rng, run_dir, write_run_config
from src.common.prompts import INSTRUCTION, block_flags, build_prefill_prompt
from src.common.sampling import reader_sampling, split_sampling

SUMMARY = REPO_ROOT / "data/screen_summary.csv"
SUMMARY_COLS = ["reader", "problem_id", "n", "n_answered", "n_correct", "p_correct", "n_capped",
                "n_reopened", "band_low", "band_high", "in_band"]


def assert_template_prefix(client, model, question, prompt):
    """The server's own chat-template render must be a token prefix of our prompt."""
    chat = client.tokenize_chat([{"role": "user", "content": INSTRUCTION.format(problem=question)}])
    ours = client.tokenize_text(prompt)
    assert ours[: len(chat)] == chat, "server template render is NOT a prefix of our prompt"
    return f"server chat render ({len(chat)} tokens) is a prefix of our prompt ({len(ours)} tokens)"


def screen_problem(client, args, cfg, samp, trace, done, rows_path, lock):
    """Generate the missing samples for one problem; write each row immediately."""
    n_needed = cfg["saturation_screen"]["n_samples"] - done
    if n_needed <= 0:
        return 0
    prompt = build_prefill_prompt(args.reader, trace.question, [])
    seed = rng(cfg["seed"], "screen", args.reader, trace.problem_id, done).getrandbits(31)
    T, top_p, cap, extra = split_sampling(samp)
    for i, c in enumerate(client.complete(prompt, n_needed, cap, T, top_p, seed=seed, **extra)):
        answer = extract_answer(c["text"])
        row = {"run_id": args.run_id, "phase": "screen", "model": args.reader, "problem_id": trace.problem_id,
               "sample_idx": done + i, "seed": seed, "prompt": prompt, "completion": c["text"],
               "finish_reason": c["finish_reason"], "tokens": c["tokens"],
               **block_flags(args.reader, c["text"], c["finish_reason"]),
               "answer": answer, "correct": is_correct(answer, trace.gt_answer),
               "ts": dt.datetime.now(dt.UTC).isoformat()}
        with lock:
            append_row(rows_path, row)
    return n_needed


def summarise(reader, rows, traces, scr):
    out = []
    for t in traces:
        rs = [r for r in rows if r["problem_id"] == t.problem_id]
        pc = p_correct(rs)
        p = pc["p_correct"]
        out.append({"reader": reader, "problem_id": t.problem_id, "n": len(rs),
                    "n_answered": pc["n_answered"], "n_correct": sum(bool(r["correct"]) for r in rs if r.get("answer")),
                    "p_correct": None if p is None else round(p, 4),
                    "n_capped": sum(bool(r["capped"]) for r in rs), "n_reopened": sum(bool(r["reopened"]) for r in rs),
                    "band_low": scr["band_low"], "band_high": scr["band_high"],
                    "in_band": p is not None and scr["band_low"] <= p <= scr["band_high"]})
    return out


def write_summary(reader, new_rows):
    """data/screen_summary.csv: this reader's rows replace its earlier ones.
    Locked and written via a temp file + rename, because several readers'
    screens run at once and two finishing together corrupted the file
    ."""
    import fcntl, os
    with open(SUMMARY.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = []
        if SUMMARY.exists():
            with open(SUMMARY) as f:
                old = [r for r in csv.DictReader(f) if r["reader"] != reader]
        tmp = SUMMARY.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, lineterminator="\n")
            w.writeheader()
            for r in sorted(old + new_rows, key=lambda r: (r["reader"], int(r["problem_id"].split("_")[1]))):
                w.writerow({k: r.get(k) for k in SUMMARY_COLS})
        os.replace(tmp, SUMMARY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--workers", type=int, default=4, help="problems generated concurrently (one request of n each)")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()
    args.run_id = args.run_id or f"screen_{args.reader}"
    cfg = load_config(args.config)
    scr = cfg["saturation_screen"]
    samp = reader_sampling(cfg, args.reader)
    traces = load_traces(["correct_base_solution"])          # the 20 problems, one Trace each
    assert len(traces) == scr["n_problems"], f"{len(traces)} problems on disk, config says {scr['n_problems']}"
    rows_path = run_dir(args.run_id) / "rows.jsonl"

    client = VLLMClient(args.reader, timeout=args.timeout)
    client.assert_serving()
    check = assert_template_prefix(client, args.reader, traces[0].question,
                                   build_prefill_prompt(args.reader, traces[0].question, []))
    write_run_config(args.run_id, cfg, extra={"script": "03_screen_readers", "args": vars(args),
                                              "sampling_used": samp, "prompt_check": check})
    done = {}
    for r in read_rows(rows_path):
        done[r["problem_id"]] = done.get(r["problem_id"], 0) + 1
    print(f"reader {args.reader} | {len(traces)} problems x {scr['n_samples']} samples | sampling {samp} | "
          f"resuming with {sum(done.values())} rows on disk | {check}", flush=True)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(screen_problem, client, args, cfg, samp, t, done.get(t.problem_id, 0), rows_path, lock): t
                for t in traces}
        for fut in as_completed(futs):
            t = futs[fut]
            print(f"{dt.datetime.now(dt.UTC):%H:%M:%S}  {t.problem_id}: +{fut.result()} rows", flush=True)

    rows = list(read_rows(rows_path))
    summary = summarise(args.reader, rows, traces, scr)
    write_summary(args.reader, summary)
    print(f"\n{'problem':<14}{'n':>4}{'answered':>10}{'correct':>9}{'P(correct)':>12}{'capped':>8}{'reopened':>10}  in band")
    for s in summary:
        p = "  -  " if s["p_correct"] is None else f"{s['p_correct']:.2f}"
        print(f"{s['problem_id']:<14}{s['n']:>4}{s['n_answered']:>10}{s['n_correct']:>9}{p:>12}"
              f"{s['n_capped']:>8}{s['n_reopened']:>10}  {'yes' if s['in_band'] else '-'}")
    in_band = [s["problem_id"] for s in summary if s["in_band"]]
    print(f"\nSUMMARY {args.reader}: rows={len(rows)} in_band={len(in_band)}/{len(summary)} "
          f"(need >= {scr['min_surviving_problems']}): {in_band}\n-> {SUMMARY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
