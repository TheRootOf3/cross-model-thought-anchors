#!/usr/bin/env python
"""Calibration: run the Phase 1 measurement with
A itself as the reader on one selected trace's three anchors, and compare our
importances with the released ones.

    python scripts/05_calibrate_on_A.py [--config configs/phase1.json] [--run_id h4_calibrate_A]
        [--trace correct_base_solution/problem_4019] [--workers 6] [--timeout 7200] [--ack]

Piles per anchor: kept (sentences 1..i prefilled, 40) and replaced (1..i-1
prefilled, 60, semantic filter, top-ups), A's own sampling (T 0.6 / top-p
0.95) at the pre-registered design cap, from src/phase1/piles. Our importance = P(correct |
replaced) − P(correct | kept), answered-only, re-opened rows excluded; the
released value is counterfactual_importance_accuracy, whose kept side is the
released accuracy at sentence i+1. Before sampling, the kept and replaced
prompts are asserted equal to BOS + the dataset's stored prompts at those positions.
The trace defaults to a seeded pick from data/phase1_traces.json (written to
data/h4_calibration_trace.json). CHECKPOINT: same ordering of the three,
magnitudes within ~2x? Rerun with --ack to record the judgement.
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.client import VLLMClient
from src.common.data import ROLLOUTS_ROOT, Trace
from src.common.filters import SemanticFilter
from src.common.io import REPO_ROOT, load_config, rng, run_dir, write_run_config
from src.common.prompts import A_BOS
from src.common.sampling import reader_sampling
from src.phase1.piles import PILES, ensure_pile, importance, pile_prompt

A = "r1-distill-14b"
TRACE_FILE = REPO_ROOT / "data/h4_calibration_trace.json"


def pick_trace(cfg) -> str:
    sel = json.load(open(REPO_ROOT / "data/phase1_traces.json"))["traces"]
    ids = sorted(f"{t['split']}/{t['problem_id']}" for t in sel)
    choice = rng(cfg["seed"], "h4_calibration_trace").choice(ids)
    TRACE_FILE.write_text(json.dumps({"seed": cfg["seed"], "salt": "h4_calibration_trace", "candidates": ids, "trace": choice}, indent=2))
    return choice


def assert_dataset_prompt(trace, i, pile, prompt):
    """A's prompt for this pile must equal BOS + the dataset's stored prompt at
    the matching position (kept at i = stored chunk i+1, replaced at i = chunk
    i); the stored text has no BOS because the API added the token itself."""
    pos = i + 1 if pile == "kept" else i
    if not trace.prefix_matches_dataset(pos):
        return f"chunk {pos}: corrupted-prefix position in the release, not compared"
    stored = trace.rollouts(pos)[0]
    stored_prompt = stored["full_cot"][: len(stored["full_cot"]) - len(stored["rollout"])]
    assert prompt == A_BOS + stored_prompt, f"{pile}@{i}: our prompt != BOS + dataset's stored prompt at chunk {pos}"
    return f"chunk {pos}: equals BOS + the dataset's stored prompt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default="h4_calibrate_A")
    ap.add_argument("--trace", default=None, help="split/problem_id; default: seeded pick from data/phase1_traces.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--ack", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    trace_id = args.trace or pick_trace(cfg)
    split, pid = trace_id.split("/")
    trace = Trace(ROLLOUTS_ROOT / split / pid, split)
    pairs = next(t for t in json.load(open(REPO_ROOT / "data/phase1_traces.json"))["traces"]
                 if t["split"] == split and t["problem_id"] == pid)["pairs"]
    samp = reader_sampling(cfg, A)
    rows_path = run_dir(args.run_id) / "rows.jsonl"

    client = VLLMClient(A, timeout=args.timeout)
    client.assert_serving()
    toks = client.tokenize_text(pile_prompt(A, trace, pairs[0]["anchor_chunk"], "kept"))
    assert toks[0] == 151646 and toks.count(151646) == 1, f"A's BOS not first and once in the server's tokenisation: {toks[:3]}"
    checks = {f"{p['anchor_chunk']}/{pile}": assert_dataset_prompt(trace, p["anchor_chunk"], pile, pile_prompt(A, trace, p["anchor_chunk"], pile))
              for p in pairs for pile in PILES}
    write_run_config(args.run_id, cfg, extra={"script": "05_calibrate_on_A", "args": vars(args), "trace": trace_id,
                                              "anchors": [p["anchor_chunk"] for p in pairs], "sampling_used": samp,
                                              "prompt_checks": checks, "bos_token_check": {"first_id": toks[0], "count": toks.count(151646)}})
    print(f"trace {trace_id} ({trace.n_chunks} sentences) | anchors {[p['anchor_chunk'] for p in pairs]} | reader {A} {samp}")
    for k, v in checks.items():
        print(f"  prompt check {k}: {v}")

    imp_cfg = cfg["importance"]
    filt = SemanticFilter(imp_cfg["similarity_model"], imp_cfg["similarity_device"], imp_cfg["similarity_max_cosine"])
    lock = threading.Lock()
    jobs = [(p["anchor_chunk"], pile) for p in pairs for pile in PILES]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ensure_pile, client, cfg, A, trace, i, pile, samp, rows_path, args.run_id, "calibration", filt, lock): (i, pile)
                for i, pile in jobs}
        piles = {futs[f]: f.result() for f in futs}

    print(f"\n{'anchor':>7} {'dir':>4} | {'released cf-imp':>15} {'acc(i+1)':>9} | {'ours':>7} {'p_kept':>7} {'p_repl':>7} | {'kept n/usable/ans':>18} {'repl n/dissim/ans':>18} | ratio")
    results = []
    for p in pairs:
        i = p["anchor_chunk"]
        res = importance(piles[(i, "kept")], piles[(i, "replaced")])
        k, r = res["kept"], res["replaced"]
        released = p["anchor_cf_importance"]
        acc_next = trace.chunks_labeled[i + 1]["accuracy"]
        ratio = None if res["importance"] is None or released == 0 else res["importance"] / released
        results.append({"anchor_chunk": i, "released": released, "ours": res["importance"], "ratio": ratio, **{f"kept_{a}": b for a, b in k.items()}, **{f"replaced_{a}": b for a, b in r.items()}})
        short = [pile for pile, st in (("kept", k), ("replaced", r)) if st["n_answered"] < imp_cfg["min_answered_per_pile"]]
        if short:
            print(f"  NOTE anchor {i}: {', '.join(short)} pile short of {imp_cfg['min_answered_per_pile']} answered usable rollouts after its top-ups (reported, not averaged in silently)")
        ours = "  undef" if res["importance"] is None else f"{res['importance']:+.3f}"
        pk = "  undef" if k["p_correct"] is None else f"{k['p_correct']:.3f}"
        pr = "  undef" if r["p_correct"] is None else f"{r['p_correct']:.3f}"
        print(f"{i:>7} {'+' if p['anchor_direction'] > 0 else '-':>4} | {released:>+15.3f} {acc_next:>9.3f} | {ours:>7} {pk:>7} {pr:>7} | "
              f"{k['n']:>5}/{k['n_usable']:>3}/{k['n_answered']:>3}       {r['n']:>5}/{r['n_dissimilar']:>3}/{r['n_answered']:>3}       | "
              f"{'-' if ratio is None else f'{ratio:.2f}'}")
    order_released = [x["anchor_chunk"] for x in sorted(results, key=lambda x: x["released"])]
    order_ours = [x["anchor_chunk"] for x in sorted(results, key=lambda x: (x["ours"] if x["ours"] is not None else 0))]
    within = [x["anchor_chunk"] for x in results if x["ratio"] is not None and 0.5 <= x["ratio"] <= 2.0]
    print(f"\nordering by signed importance: released {order_released} | ours {order_ours} | same: {order_released == order_ours}")
    print(f"magnitude within 2x (0.5 <= ours/released <= 2): {len(within)}/3 {within}")
    (run_dir(args.run_id) / "summary.json").write_text(json.dumps({"trace": trace_id, "results": results, "order_same": order_released == order_ours,
                                                                    "within_2x": within}, indent=2))
    if not args.ack:
        print("\nCHECKPOINT (the calibration run): does our ordering match the released one for the three anchors, with magnitudes within ~2x? "
              "If not, fix prompts/extraction (45-min box) or log the discrepancy and proceed flagged. Rerun with --ack to record.")
        sys.exit(2)
    (run_dir(args.run_id) / "ack.txt").write_text(__import__("time").strftime("%Y-%m-%d %H:%M UTC\n", __import__("time").gmtime()))
    print("acknowledged")


if __name__ == "__main__":
    main()
