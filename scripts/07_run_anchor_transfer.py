#!/usr/bin/env python
"""The Phase 1 grid for one reader: reader importance at every anchor and partner.

    python scripts/07_run_anchor_transfer.py --reader gpt-oss-20b [--config configs/phase1.json]
        [--run_id p1_<reader>] [--workers 10] [--timeout 7200]

For every trace in the reader's list (data/phase1_traces.json `by_reader`),
every anchor/partner pair, both sentences, both piles: src/phase1/piles
.ensure_pile - kept 40, replaced 60 with the semantic filter, the pre-registered design top-ups -
with the reader's own sampling (configs/models.json) at the pre-registered design cap. Rows go
to runs/<run_id>/rows.jsonl as they land; a rerun resumes per (trace,
sentence, pile), refuses rows made with a different prompt, and refuses to
resume under a changed config, sampling or trace list. Jobs are queued by
pair rank (every trace's rank-1 pair first) so a complete subset forms
early for 08. A failed pile is reported and the rest continue; the script
exits 1 at the end naming the failures (rerun to resume them). Prints each
finished pile and, at the end, per pair: importance_B at the anchor and
partner and d = sign_A x (anchor - partner) (src/common/stats.paired_d);
summary.json holds the same. CIs and the gate are 08_analyze_anchors.py.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.client import VLLMClient
from src.common.data import ROLLOUTS_ROOT, Trace
from src.common.filters import SemanticFilter
from src.common.io import REPO_ROOT, load_config, read_rows, run_dir, write_run_config
from src.common.prompts import pin_render_date, render_date
from src.common.sampling import reader_sampling
from src.common.stats import paired_d
from src.phase1.piles import PILES, ensure_pile, importance, pile_stats, usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_id = args.run_id or f"p1_{args.reader}"
    sel = json.load(open(REPO_ROOT / "data/phase1_traces.json"))
    traces = sel["by_reader"][args.reader]["traces"]
    existing_rows = list(read_rows(run_dir(run_id, create=True) / "rows.jsonl"))
    date = next((d for r in existing_rows if (d := render_date(r["prompt"]))), None)
    pin_render_date(date)          # extend rows with the prompt they were made with, not today's render
    samp = reader_sampling(cfg, args.reader)
    rows_path = run_dir(run_id) / "rows.jsonl"
    client = VLLMClient(args.reader, timeout=args.timeout)
    client.assert_serving()
    write_run_config(run_id, cfg, extra={"script": "07_run_anchor_transfer", "args": vars(args), "sampling_used": samp,
                                         "traces": [f"{t['split']}/{t['problem_id']}" for t in traces],
                                         "render_date": date,
                                         "pairs": [f"{t['split']}/{t['problem_id']}#{p.get('anchor_set', 'matched')}{p['anchor_rank']}" for t in traces for p in t["pairs"]]})
    stored = json.load(open(run_dir(run_id) / "config.json"))       # on resume: the run must mean what it meant
    assert stored["importance"] == cfg["importance"] and stored["_run"]["sampling_used"] == samp \
        and stored["_run"]["traces"] == [f"{t['split']}/{t['problem_id']}" for t in traces] \
        and stored["_run"].get("pairs", None) in (None, [f"{t['split']}/{t['problem_id']}#{p.get('anchor_set', 'matched')}{p['anchor_rank']}" for t in traces for p in t["pairs"]]) \
        and stored["_run"].get("render_date", date) == date, \
        f"runs/{run_id}/config.json differs from the current config, sampling, trace list, pair list or render date; use a new run_id"
    foreign = [r for r in read_rows(rows_path) if r["run_id"] != run_id or r["reader"] != args.reader or r["phase"] != "p1"]
    assert not foreign, f"runs/{run_id}/rows.jsonl holds {len(foreign)} rows from another run/reader/phase"
    loaded = {f"{t['split']}/{t['problem_id']}": Trace(ROLLOUTS_ROOT / t["split"] / t["problem_id"], t["split"]) for t in traces}
    jobs = [(t, p, role, pile)                                   # rank-major: all rank-1 pairs first
            for rank in (1, 2, 3) for t in traces for p in t["pairs"] if p["anchor_rank"] == rank
            for role in ("anchor", "partner") for pile in PILES]
    print(f"{args.reader}: {len(traces)} traces, {sum(len(t['pairs']) for t in traces)} pairs, {len(jobs)} piles | {samp} | "
          f"kept {cfg['importance']['kept_samples']}, replaced {cfg['importance']['replaced_samples']} + top-ups -> {rows_path.relative_to(REPO_ROOT)}")

    imp = cfg["importance"]
    filt = SemanticFilter(imp["similarity_model"], imp["similarity_device"], imp["similarity_max_cosine"])
    lock = threading.Lock()

    def run(t, p, role, pile):
        trace = loaded[f"{t['split']}/{t['problem_id']}"]
        i = p[f"{role}_chunk"]
        return ensure_pile(client, cfg, args.reader, trace, i, pile, samp, rows_path, run_id, "p1", filt, lock)

    t0 = time.time()
    piles, failed = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, *j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            t, p, role, pile = futs[f]
            try:
                rows = f.result()
            except Exception as e:                       # one pile's failure must not hide the others
                failed.append(f"{t['split']}/{t['problem_id']} {p.get('anchor_set', 'matched')} pair {p['anchor_rank']} {role} {pile}")
                print(f"[{n:>3}/{len(jobs)} {time.time() - t0:>6.0f}s] FAILED {failed[-1]}: {e!r}", flush=True)
                continue
            piles[(t["split"], t["problem_id"], p.get("anchor_set", "matched"), p["anchor_rank"], role, pile)] = rows
            st = pile_stats(rows)
            print(f"[{n:>3}/{len(jobs)} {time.time() - t0:>6.0f}s] {t['split'][:3]}/{t['problem_id']} {p.get('anchor_set', 'matched')[:3]} pair {p['anchor_rank']} {role:<7} {pile:<8} "
                  f"n {st['n']:>3} usable {st['n_usable']:>3} answered {st['n_answered']:>3} p {'-' if st['p_correct'] is None else f'{st['p_correct']:.2f}'}"
                  + (f" dissimilar {st['n_dissimilar']}" if pile == "replaced" else "") + (f" reopened {st['n_reopened']}" if st["n_reopened"] else ""), flush=True)

    if failed:
        print(f"\n{len(failed)} of {len(jobs)} piles failed; rerun the same command to resume them:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print(f"\n{'trace':<36} {'set':>8} {'pair':>4} {'anchor':>6} {'imp_A':>6} {'imp_B(anchor)':>13} {'imp_B(partner)':>14} {'d':>7}")
    summary = []
    for t in traces:
        for p in t["pairs"]:
            k = (t["split"], t["problem_id"], p.get("anchor_set", "matched"), p["anchor_rank"])
            a = tuple(usable(piles[k + ("anchor", pile)]) for pile in PILES)
            q = tuple(usable(piles[k + ("partner", pile)]) for pile in PILES)
            d = paired_d(a, q, p["anchor_direction"])
            ia, ip = importance(*a)["importance"], importance(*q)["importance"]
            summary.append({"trace": f"{t['split']}/{t['problem_id']}", "anchor_set": p.get("anchor_set", "matched"), "anchor_rank": p["anchor_rank"], "anchor_chunk": p["anchor_chunk"],
                            "partner_chunk": p["partner_chunk"], "direction": p["anchor_direction"], "importance_A": p["anchor_cf_importance"],
                            "importance_B_anchor": ia, "importance_B_partner": ip, "d": d,
                            "stats": {f"{role}_{pile}": pile_stats(piles[k + (role, pile)]) for role in ("anchor", "partner") for pile in PILES}})
            fmt = lambda v: "  undef" if v is None else f"{v:+.3f}"
            print(f"{t['split'][:3] + '/' + t['problem_id']:<36} {p.get('anchor_set', 'matched'):>8} {p['anchor_rank']:>4} {p['anchor_chunk']:>6} {p['anchor_cf_importance']:>+6.2f} {fmt(ia):>13} {fmt(ip):>14} {fmt(d):>7}")
    ds = [s["d"] for s in summary if s["d"] is not None]
    print(f"\n{len(ds)} pairs with d defined" + (f"; mean d {sum(ds) / len(ds):+.3f}; d > 0 in {sum(x > 0 for x in ds)}/{len(ds)}" if ds else "")
          + "  (hypothesis - CIs, permutation p and the gate are 08_analyze_anchors.py)")
    (run_dir(run_id) / "summary.json").write_text(json.dumps({"reader": args.reader, "pairs": summary}, indent=2))


if __name__ == "__main__":
    main()
