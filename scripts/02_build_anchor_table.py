"""Build the anchor table: every sentence of every released
trace with A's released scores (counterfactual and resampling importance, accuracy
and KL forms, under their dataset field names), plus the anchor/partner assignment.

Rules (configs/phase1.json trace_selection), fixed before any reader rollout:
- testable sentence = before the convergence index (thought-anchors k=4 rule),
  released importance measured, overdeterminedness (1 - different_trajectories_
  fraction, the paper's quantity) <= max_overdeterminedness (its plot filter),
  stored rollouts at i AND i+1 from the true prefix (src/common/data.py Trace.testable);
- anchors = top-3 testable sentences by |released counterfactual importance
  (accuracy)|, ties by lower index, in the direction matching the trace
  outcome (anchor_direction_rule "match_trace_outcome": negative on correct
  traces, positive on wrong traces; "either" = direction-free; "both" = the
  matched top-3 exactly as above and then the top-3 of the OPPOSITE direction
  among the sentences not yet used, extension of 2026-09-10, column
  anchor_set matched/opposite); each anchor's direction is recorded and
  orients the paired d;
- partner = the testable, non-anchor, not-yet-used sentence within +/-4 whose
  SIGNED importance is closest to zero (ties: nearer, then lower index);
  no partner -> the anchor is dropped. decided 2026-09-08 (the anchor table).

Writes data/anchor_table.csv (one row per sentence, all 40 traces) and
data/h2_checkpoint_traces.json (the 5 traces printed, seeded).

CHECKPOINT: prints the anchors and partners of 5 traces and exits.
Rerun with --ack after reading them.

Usage: python scripts/02_build_anchor_table.py --config configs/phase1.json --run_id h2_anchor_table [--ack]
       python scripts/02_build_anchor_table.py --rule both --out data/anchor_table_ext.csv --run_id h2_anchor_table_ext [--ack]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data import assert_dataset_present, Trace, load_traces
from src.common.io import REPO_ROOT, load_config, rng, run_dir, write_run_config

TABLE = REPO_ROOT / "data/anchor_table.csv"
CHECKPOINT_TRACES = REPO_ROOT / "data/h2_checkpoint_traces.json"
COLUMNS = ["split", "problem_id", "base_correct", "chunk_idx", "n_chunks", "text", "tags",
           "accuracy", "counterfactual_importance_accuracy", "counterfactual_importance_kl",
           "resampling_importance_accuracy", "resampling_importance_kl", "overdeterminedness", "measured",
           "not_overdetermined", "prefix_ok", "convergence_idx", "testable",
           "anchor_rank", "anchor_direction", "anchor_set", "anchor_dropped", "partner_of", "partner_distance"]


def assign(trace: Trace, ts: dict, rule: str | None = None) -> dict[int, dict]:
    """chunk_idx -> anchor/partner fields for one trace. Sets are assigned in order,
    each from the sentences the earlier sets did not use, so "both" reproduces
    the matched set of "match_trace_outcome" exactly and adds the opposite one."""
    rule = rule or ts.get("anchor_direction_rule", "either")
    testable = [i for i in range(trace.n_chunks)
                if trace.testable(i, k=ts["convergence_k"], max_overdeterminedness=ts["max_overdeterminedness"])]
    imp = {i: trace.importance_a(i) for i in testable}
    want = -1 if trace.base_is_correct else 1              # correct trace: A needed it; wrong trace: A better without it
    sets = {"match_trace_outcome": [("matched", want)], "either": [("either", 0)], "both": [("matched", want), ("opposite", -want)]}[rule]
    used, out = set(), {}
    for name, sign in sets:
        candidates = [i for i in testable if i not in used and (imp[i] * sign > 0 if sign else imp[i] != 0)]
        anchors = sorted(candidates, key=lambda i: (-abs(imp[i]), i))[: ts["anchors_per_trace"]]
        used |= set(anchors)
        for rank, a in enumerate(anchors, 1):
            cands = [j for j in testable if j not in used and abs(j - a) <= ts["partner_window"]]
            if not cands:
                out[a] = {"anchor_rank": rank, "anchor_direction": 1 if imp[a] > 0 else -1, "anchor_set": name, "anchor_dropped": True}
                continue
            p = min(cands, key=lambda j: (abs(imp[j]), abs(j - a), j))
            used.add(p)
            out[a] = {"anchor_rank": rank, "anchor_direction": 1 if imp[a] > 0 else -1, "anchor_set": name, "anchor_dropped": False}
            out[p] = {"partner_of": a, "partner_distance": abs(p - a)}
    return out


def rows_for(trace: Trace, ts: dict, rule: str | None = None) -> list[dict]:
    k, max_od = ts["convergence_k"], ts["max_overdeterminedness"]
    conv = trace.convergence_index(k)
    roles = assign(trace, ts, rule)
    rows = []
    for i, (text, lab) in enumerate(zip(trace.chunks, trace.chunks_labeled)):
        prefix_ok = (i + 1 < trace.n_chunks and trace.prefix_matches_dataset(i)
                     and trace.prefix_matches_dataset(i + 1))
        rows.append({
            "split": trace.split, "problem_id": trace.problem_id,
            "base_correct": trace.base_is_correct, "chunk_idx": i, "n_chunks": trace.n_chunks,
            "text": text, "tags": "|".join(lab.get("function_tags", [])),
            "accuracy": lab.get("accuracy"),
            "counterfactual_importance_accuracy": lab.get("counterfactual_importance_accuracy"),
            "counterfactual_importance_kl": lab.get("counterfactual_importance_kl"),
            "resampling_importance_accuracy": lab.get("resampling_importance_accuracy"),
            "resampling_importance_kl": lab.get("resampling_importance_kl"),
            "overdeterminedness": trace.overdeterminedness(i),
            "measured": trace.is_measured(i), "not_overdetermined": trace.not_overdetermined(i, max_od),
            "prefix_ok": prefix_ok, "convergence_idx": conv,
            "testable": trace.testable(i, k=k, max_overdeterminedness=max_od),
            **roles.get(i, {}),
        })
    return rows


def print_checkpoint(rows: list[dict], trace_ids: list[str]) -> None:
    by_trace = {}
    for r in rows:
        by_trace.setdefault((r["split"], r["problem_id"]), []).append(r)
    for key in by_trace:
        if f"{key[0]}/{key[1]}" not in trace_ids:
            continue
        rs = by_trace[key]
        print(f"\n=== {key[0]}/{key[1]}  base_correct={rs[0]['base_correct']}  "
              f"n_chunks={rs[0]['n_chunks']}  convergence_idx={rs[0]['convergence_idx']}  "
              f"testable={sum(r['testable'] for r in rs)}")
        for a in sorted((r for r in rs if r.get("anchor_rank")), key=lambda r: (r.get("anchor_set") != "matched", r["anchor_rank"])):
            direction = "replacing it HELPED A" if a["anchor_direction"] > 0 else "replacing it HURT A"
            print(f"\n  ANCHOR #{a['anchor_rank']} ({a.get('anchor_set')})  chunk {a['chunk_idx']}  cf-imp {a['counterfactual_importance_accuracy']:+.3f} ({direction})  "
                  f"acc {a['accuracy']:.2f}  overdet {a['overdeterminedness']:.2f}  [{a['tags']}]"
                  + ("  DROPPED (no partner)" if a["anchor_dropped"] else ""))
            print(f"    {a['text']}")
            for p in (r for r in rs if r.get("partner_of") == a["chunk_idx"]):
                print(f"  partner  chunk {p['chunk_idx']}  (dist {p['partner_distance']})  cf-imp {p['counterfactual_importance_accuracy']:+.3f}  "
                      f"acc {p['accuracy']:.2f}  [{p['tags']}]")
                print(f"    {p['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default="h2_anchor_table")
    ap.add_argument("--ack", action="store_true", help="the checkpoint has been read")
    ap.add_argument("--rule", default=None, choices=["match_trace_outcome", "either", "both"], help="override the config's anchor_direction_rule")
    ap.add_argument("--out", default=None, help="table path (default data/anchor_table.csv)")
    args = ap.parse_args()
    assert_dataset_present()      # the released rollouts, not our data - see the README's Setup
    cfg = load_config(args.config)
    ts = cfg["trace_selection"]
    table = REPO_ROOT / args.out if args.out else TABLE
    write_run_config(args.run_id, cfg, extra={"script": "02_build_anchor_table", "args": vars(args), "rule": args.rule or ts.get("anchor_direction_rule"), "table": str(table.relative_to(REPO_ROOT))})

    traces = load_traces(cfg["dataset"]["splits"])
    rows = [r for t in traces for r in rows_for(t, ts, args.rule)]
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    anchors = [r for r in rows if r.get("anchor_rank")]
    dropped = [(r["split"][:3], r["problem_id"], r["chunk_idx"]) for r in anchors if r["anchor_dropped"]]
    pairs_per_trace = {(t.split[:3], t.problem_id): 0 for t in traces}
    for r in anchors:
        if not r["anchor_dropped"]:
            pairs_per_trace[(r["split"][:3], r["problem_id"])] += 1
    per_set = len({r.get("anchor_set") for r in anchors})
    short = [t for t in pairs_per_trace if pairs_per_trace[t] < ts["anchors_per_trace"] * per_set]
    print(f"traces {len(traces)}  rows {len(rows)}  testable {sum(r['testable'] for r in rows)}  "
          f"never-converged {sum(r['chunk_idx'] == 0 and r['convergence_idx'] is None for r in rows)}")
    print(f"pairs {sum(pairs_per_trace.values())}  anchors with direction +1/-1: "
          f"{sum(r['anchor_direction'] > 0 for r in anchors)}/{sum(r['anchor_direction'] < 0 for r in anchors)}  "
          f"dropped (no partner within window) {dropped}  traces with <{ts['anchors_per_trace']} pairs {short}")
    print(f"wrote {table}")

    ids = sorted(f"{t.split}/{t.problem_id}" for t in traces)
    picked = rng(cfg["seed"], "h2_checkpoint").sample(ids, 5)
    CHECKPOINT_TRACES.write_text(json.dumps({"seed": cfg["seed"], "salt": "h2_checkpoint", "traces": picked}, indent=2))
    print_checkpoint(rows, picked)

    if not args.ack:
        print("\nCHECKPOINT (the anchor table): read the anchors and partners above. Do the anchors read as the "
              "sentences that decided A's outcome (the step that locked in the error, or the step A needed), "
              "and the partners as low-stakes? Rerun with --ack to record that you have read them.")
        sys.exit(2)
    (run_dir(args.run_id) / "ack.txt").write_text(time.strftime("%Y-%m-%d %H:%M UTC\n", time.gmtime()))
    print("acknowledged")


if __name__ == "__main__":
    main()
