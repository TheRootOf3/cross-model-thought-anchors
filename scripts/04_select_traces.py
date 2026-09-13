#!/usr/bin/env python
"""Trace selection: the traces and anchor/partner pairs each reader is measured on.

    python scripts/04_select_traces.py [--config configs/phase1.json] [--run_id h3_select_traces]
        [--readers qwen3-1.7b gpt-oss-20b]

Readers: by default the pre-registered design rule applied to data/screen_summary.csv - B_near is
Qwen3.5-9B if it has >= min_surviving_problems in band, else Qwen3.5-4B; B_far
is gpt-oss-20b if in band, else Olmo (configs/models.json roles). Trace sets
(config trace_sets): 'per_reader' (2026-09-09) - each reader draws 5 correct-A
+ 5 incorrect-A traces from 10 DISJOINT problems in ITS band (15-85%,
answered-only) with the seeded RNG (salt includes the reader); 'shared' - one
set from problems in band for both. Each trace's 3 anchors and partners come
from data/anchor_table.csv. Writes data/phase1_traces.json (`by_reader`, the
shared problems, and `traces` = B_far's list) and prints the tables. Stops
(exit 3) if a reader has fewer than 10 in band - the author decides.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import REPO_ROOT, load_config, rng, write_run_config

SCREEN = REPO_ROOT / "data/screen_summary.csv"
TABLE = REPO_ROOT / "data/anchor_table.csv"
OUT = REPO_ROOT / "data/phase1_traces.json"


def screen_rows():
    with open(SCREEN) as f:
        return [r for r in csv.DictReader(f)]


def in_band(rows, reader):
    return sorted({r["problem_id"] for r in rows if r["reader"] == reader and r["in_band"] == "True"},
                  key=lambda p: int(p.split("_")[1]))


def pick_readers(rows, roles, min_problems):
    """The two readers named in configs/models.json roles, each checked against the screen.

    Both must clear min_problems problems inside the accuracy band. The same-family role has no
    fallback left - the candidates that were meant to be its fallback saturated at 1 of 20 problems
    in band - so a reader that misses the bar is a stop, not a silent substitution.
    """
    near = roles["B_near"]
    if len(in_band(rows, near)) < min_problems:
        raise SystemExit(f"STOP: {near} has only {len(in_band(rows, near))} problems in band, "
                         f"need {min_problems}; no fallback is configured. Screen another reader.")
    far = roles["B_far"]
    if len(in_band(rows, far)) < min_problems:
        far = next(c for c in roles["B_far_candidates"] if c != roles["B_far"])
    return near, far


def pairs_for(table, split, problem_id):
    rows = [r for r in table if r["split"] == split and r["problem_id"] == problem_id]
    by_idx = {int(r["chunk_idx"]): r for r in rows}
    out = []
    for a in sorted((r for r in rows if r["anchor_rank"]), key=lambda r: (r.get("anchor_set", "matched") != "matched", int(r["anchor_rank"]))):
        if a["anchor_dropped"] == "True":
            continue
        p = next(r for r in rows if r["partner_of"] and int(r["partner_of"]) == int(a["chunk_idx"]))
        out.append({"anchor_rank": int(a["anchor_rank"]), "anchor_set": a.get("anchor_set") or "matched", "anchor_chunk": int(a["chunk_idx"]),
                    "anchor_direction": int(a["anchor_direction"]),
                    "anchor_cf_importance": float(a["counterfactual_importance_accuracy"]),
                    "anchor_text": a["text"], "anchor_tags": a["tags"],
                    "partner_chunk": int(p["chunk_idx"]), "partner_distance": int(p["partner_distance"]),
                    "partner_cf_importance": float(p["counterfactual_importance_accuracy"]),
                    "partner_text": p["text"], "partner_tags": p["tags"]})
    return out, len(by_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default="h3_select_traces")
    ap.add_argument("--readers", nargs=2, default=None, help="override the the pre-registered design rule: B_near B_far")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ts, scr = cfg["trace_selection"], cfg["saturation_screen"]
    roles = load_config(REPO_ROOT / "configs/models.json")["roles"]
    rows = screen_rows()
    readers_screened = sorted({r["reader"] for r in rows})
    near, far = args.readers or pick_readers(rows, roles, scr["min_surviving_problems"])
    for r in readers_screened:
        print(f"{r:<16} in band on {len(in_band(rows, r)):>2}/{scr['n_problems']} problems: {in_band(rows, r)}")
    print(f"readers: B_near = {near}, B_far = {far}  ({'--readers override' if args.readers else 'the pre-registered design rule'})")
    eligible = sorted(set(in_band(rows, near)) & set(in_band(rows, far)), key=lambda p: int(p.split("_")[1]))
    print(f"eligible (in band for both): {len(eligible)}: {eligible}")
    write_run_config(args.run_id, cfg, extra={"script": "04_select_traces", "args": vars(args),
                                              "readers": [near, far], "eligible": eligible})
    need = ts["n_from_correct"] + ts["n_from_incorrect"]
    with open(TABLE) as f:
        table = list(csv.DictReader(f))

    def draw(pool, salt):
        picked = rng(cfg["seed"], "select_traces", *salt).sample(pool, need)
        out = []
        for split, ids in (("correct_base_solution", picked[: ts["n_from_correct"]]), ("incorrect_base_solution", picked[ts["n_from_correct"]:])):
            for pid in ids:
                pairs, n_chunks = pairs_for(table, split, pid)
                out.append({"split": split, "problem_id": pid, "n_chunks": n_chunks, "pairs": pairs})
        return out

    mode = ts.get("trace_sets", "shared")
    if mode == "shared":                         # the pre-registered design as first written: one set, in band for both
        if len(eligible) < need:
            print(f"\nSTOP: only {len(eligible)} eligible problems, need {need} disjoint ones. Stop and decide.")
            sys.exit(3)
        by_reader = {r: {"eligible": eligible, "traces": draw(eligible, (near, far))} for r in {near, far}}
    else:                                        # per_reader (changelog 2026-09-09): each reader draws from its own band
        by_reader = {}
        for r in dict.fromkeys((far, near)):     # B_far first: its list is the top-level `traces`
            pool = in_band(rows, r)
            if len(pool) < need:
                print(f"\nSTOP: {r} has only {len(pool)} problems in band, need {need}. Stop and decide.")
                sys.exit(3)
            by_reader[r] = {"eligible": pool, "traces": draw(pool, (r, r))}   # salt (r, r) = the reader in both roles: reproduces the gpt-oss-only draw of 2026-09-09
    OUT.write_text(json.dumps({"seed": cfg["seed"], "trace_sets": mode, "table": str(TABLE.relative_to(REPO_ROOT)), "readers": {"B_near": near, "B_far": far},
                               "shared_problems": eligible, "by_reader": by_reader, "traces": by_reader[far]["traces"]}, indent=2))
    for r, sel in by_reader.items():
        print(f"\n{r}: {len(sel['eligible'])} eligible, {len(sel['traces'])} traces, {sum(len(t['pairs']) for t in sel['traces'])} pairs")
        print(f"{'trace':<40}{'pairs':>6}   anchors (chunk, direction)")
        for t in sel["traces"]:
            print(f"{t['split'] + '/' + t['problem_id']:<40}{len(t['pairs']):>6}   "
                  + ", ".join(f"{p['anchor_chunk']}{'+' if p['anchor_direction'] > 0 else '-'}{'' if p['anchor_set'] == 'matched' else 'o'}" for p in t["pairs"]))
    print(f"\nshared problems (in band for both): {len(eligible)} -> {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
