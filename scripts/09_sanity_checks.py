#!/usr/bin/env python
"""Sanity checks. Each mode prints its check and appends one row
to doc/checks.md. every headline number gets an independent
recompute from the raw JSONL, in code that does not share the analysis path.

    python scripts/09_sanity_checks.py --recompute --run_id screen_<reader> # screens
    python scripts/09_sanity_checks.py [--readers gpt-oss-20b qwen3-1.7b] --all # Phase 1 (Phase 1), or any of:
        --recompute --extraction --restarts --filter --kept --position --caps --per_trace

--recompute (screens): re-extract answer/correct and re-derive the block flags
from every stored completion, recount per problem with its own code, compare
with data/screen_summary.csv.
Phase 1 modes read runs/p1_<reader>/rows.jsonl and data/phase1_traces.json:
--recompute re-extract every completion; mean d and fraction positive from
              own code (no 08 path); compare with data/p1_results.json.
--extraction 10 seeded answered rows per reader (5 correct, 5 wrong): tail of the
              completion, extracted answer, ground truth - Read.
--restarts re-open rates by pile and role, sentences with heavy re-opening;
              d recomputed WITH re-opened rollouts (sensitivity).
--filter 10 seeded replaced rows (5 kept, 5 dropped) with A's sentence, the
              reader's first sentence and the cosine - Read; survivors per
              pile; d with the filter off (sensitivity).
--kept 10 seeded kept rows: A's prefilled sentence and how the reader
              continues - Read.
--position anchor position along the trace vs d, importance_B vs position for
              anchors and partners.
--caps cap-hit and no-answer rates per pile, pairs whose piles answer at
              different rates (with --restarts).
--per_trace per-trace mean d, sign test over traces.
Each mode prints its check and appends one row to doc/checks.md. Figures:
(fig8 position, fig9 per-trace and fig10 attrition were removed 2026-09-11 at a request;
the numbers they plotted still go to doc/checks.md.)
"""

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.answers import extract_answer, is_correct
from src.common.data import ROLLOUTS_ROOT, Trace, load_traces
from src.common.io import REPO_ROOT, load_config, read_rows, rng
from src.common.prompts import block_flags

CHECKS = REPO_ROOT / "doc/checks.md"


def note(check, method, result, changed="no"):
    if not CHECKS.exists():
        CHECKS.write_text("# Sanity checks\n\n| when (UTC) | check | method | result | changed anything? |\n|---|---|---|---|---|\n")
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")     # a pipe or newline inside a cell breaks the table
    with open(CHECKS, "a") as f:
        f.write(f"| {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} | {esc(check)} | {esc(method)} | {esc(result)} | {esc(changed)} |\n")


def recompute_screen(run_id):
    rows = list(read_rows(REPO_ROOT / "runs" / run_id / "rows.jsonl"))
    assert rows, f"no rows under runs/{run_id}"
    reader = rows[0]["model"]
    gt = {t.problem_id: t.gt_answer for t in load_traces(["correct_base_solution"])}
    per, dup = {}, len(rows) - len({(r["problem_id"], r["sample_idx"]) for r in rows})
    disagree = 0
    for r in rows:
        a = extract_answer(r["completion"])
        c = is_correct(a, gt[r["problem_id"]])
        f = block_flags(reader, r["completion"], r["finish_reason"])
        disagree += (a != r["answer"]) or (c != r["correct"]) or (f["capped"] != r["capped"]) or (f["reopened"] != r["reopened"])
        d = per.setdefault(r["problem_id"], {"n": 0, "answered": 0, "correct": 0, "capped": 0, "reopened": 0})
        d["n"] += 1; d["answered"] += bool(a); d["correct"] += bool(a) and bool(c); d["capped"] += f["capped"]; d["reopened"] += f["reopened"]
    with open(REPO_ROOT / "data/screen_summary.csv") as fh:
        summ = {r["problem_id"]: r for r in csv.DictReader(fh) if r["reader"] == reader}
    mism, in_band = 0, []
    for pid, d in per.items():
        p = d["correct"] / d["answered"] if d["answered"] else None
        s = summ.get(pid)
        same = s is not None and all(int(s[k]) == d[k2] for k, k2 in (("n", "n"), ("n_answered", "answered"), ("n_correct", "correct"),
                                                                     ("n_capped", "capped"), ("n_reopened", "reopened"))) \
            and ((s["p_correct"] == "" and p is None) or (p is not None and abs(float(s["p_correct"]) - p) < 5e-5))
        mism += not same
        if p is not None and 0.15 <= p <= 0.85:
            in_band.append(pid)
    result = (f"{len(rows)} rows, {dup} duplicate keys, {disagree} rows where re-extraction disagrees, "
              f"{mism} summary rows differ; answered {sum(d['answered'] for d in per.values())}, "
              f"capped {sum(d['capped'] for d in per.values())}, reopened {sum(d['reopened'] for d in per.values())}, "
              f"in band {len(in_band)}/{len(per)}")
    print(f"{run_id}: {result}")
    note(f"screen recompute {reader}", "re-extract + recount from rows.jsonl, own code, vs data/screen_summary.csv", result)
    return dup == 0 and disagree == 0 and mism == 0


# ----------------------------------------------------------------------------- Phase 1 (Phase 1)
def _med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


class P1:
    """One reader's Phase 1 run: raw rows grouped per pile, its pairs, the traces."""

    def __init__(self, reader, cfg):
        import json
        self.reader, self.cfg = reader, cfg
        self.rows = list(read_rows(REPO_ROOT / "runs" / f"p1_{reader}" / "rows.jsonl"))
        self.by = {}
        for r in self.rows:
            self.by.setdefault((r["trace"], r["chunk_idx"], r["pile"]), []).append(r)
        self.traces = json.load(open(REPO_ROOT / "data/phase1_traces.json"))["by_reader"][reader]["traces"]
        self.trace_obj = {f"{t['split']}/{t['problem_id']}": Trace(ROLLOUTS_ROOT / t["split"] / t["problem_id"], t["split"]) for t in self.traces}
        self.pairs = [(f"{t['split']}/{t['problem_id']}", t, p) for t in self.traces for p in t["pairs"]]

    def pile(self, key, i, pile):
        return self.by.get((key, i, pile), [])

    @staticmethod
    def usable(rows, reopened_ok=False, filter_off=False):
        return [r for r in rows if (reopened_ok or not r["reopened"]) and (r["pile"] == "kept" or filter_off or r.get("dissimilar"))]

    def d_of(self, key, p, **kw):
        """d for one pair from the rows, own arithmetic (answered-only P(correct))."""
        def acc(rows):
            a = [r for r in rows if r["answer"]]
            return (sum(bool(r["correct"]) for r in a) / len(a)) if a else None
        def imp(i):
            k, r = acc(self.usable(self.pile(key, i, "kept"), **kw)), acc(self.usable(self.pile(key, i, "replaced"), **kw))
            return None if k is None or r is None else r - k
        ia, ip = imp(p["anchor_chunk"]), imp(p["partner_chunk"])
        return None if ia is None or ip is None else p["anchor_direction"] * (ia - ip)

    def readout(self, label, **kw):
        """mean d, fraction positive and the permutation p under a variant of the usable rule."""
        from src.common.stats import permutation_p
        ds, tuples = [], {}
        for key, t, p in self.pairs:
            d = self.d_of(key, p, **kw)
            if d is None:
                continue
            ds.append(d)
            a = (self.usable(self.pile(key, p["anchor_chunk"], "kept"), **kw), self.usable(self.pile(key, p["anchor_chunk"], "replaced"), **kw))
            q = (self.usable(self.pile(key, p["partner_chunk"], "kept"), **kw), self.usable(self.pile(key, p["partner_chunk"], "replaced"), **kw))
            tuples.setdefault(key, []).append((a, q, p["anchor_direction"]))
        if not ds:
            return f"{label}: no pair defined"
        perm = permutation_p(tuples, self.cfg["analysis"]["permutation_n"], rng(self.cfg["seed"], "09", label, self.reader).getrandbits(31))
        return f"{label}: mean d {sum(ds) / len(ds):+.3f}, d > 0 in {sum(x > 0 for x in ds)}/{len(ds)}, permutation p {perm['p_one_sided']:.3f}"


def p1_recompute(P):
    """Integrity + own arithmetic: the re-extraction reuses the generating functions
    (extract_answer, is_correct, block_flags), so 0 disagreements means the rows are
    intact; the duplicate-key check and the d arithmetic are independent of 08. The
    permutation p in the other modes comes from stats.permutation_p (shared with 08)."""
    import json
    dup = len(P.rows) - len({(r["run_id"], r["trace"], r["chunk_idx"], r["pile"], r["sample_idx"]) for r in P.rows})
    disagree = 0
    for r in P.rows:
        a = extract_answer(r["completion"])
        f = block_flags(P.reader, r["completion"], r["finish_reason"])
        disagree += (a != r["answer"]) or (is_correct(a, P.trace_obj[r["trace"]].gt_answer) != r["correct"]) or (f["reopened"] != r["reopened"]) or (f["capped"] != r["capped"])
    from src.phase1.piles import complete
    finished = [(key, p) for key, t, p in P.pairs if all(complete(P.pile(key, p[f"{role}_chunk"], pile), pile, P.cfg["importance"])
                                                        for role in ("anchor", "partner") for pile in ("kept", "replaced"))]
    ds = [d for key, p in finished if (d := P.d_of(key, p)) is not None]
    res_path = REPO_ROOT / "data/p1_results.json"
    res = json.load(open(res_path))["readers"].get(P.reader, {}) if res_path.exists() else {}
    ref, missing = res.get("primary_answered", {}), res.get("missing_pairs", [])
    same = bool(ds) and bool(ref) and abs(ref.get("mean_d", 9) - _mean(ds)) < 5e-4 and ref.get("n_pairs") == len(ds)
    result = (f"{len(P.rows)} rows, {dup} duplicate keys, {disagree} rows where re-extraction disagrees; own code on the {len(finished)}/{len(P.pairs)} pairs with finished piles: "
              f"n {len(ds)} with d defined, mean d {_mean(ds):+.4f}, d > 0 in {sum(x > 0 for x in ds)}/{len(ds)}; 08 reports mean d {ref.get('mean_d', float('nan')):+.4f} on {ref.get('n_pairs')}"
              + (f" ({len(missing)} pairs missing there)" if missing else "") + f" -> {'agree' if same else 'DIFFER'}")
    print(f"{P.reader} recompute: {result}")
    note(f"P1 recompute {P.reader}", "re-extract + own d from rows.jsonl vs data/p1_results.json", result)
    return dup == 0 and disagree == 0 and bool(same)


def p1_extraction(P):
    draw = rng(P.cfg["seed"], "09_extraction", P.reader)
    ans = [r for r in P.rows if r["answer"]]
    pick = lambda pool, k=5: draw.sample(pool, min(k, len(pool)))
    show = pick([r for r in ans if r["correct"]]) + pick([r for r in ans if not r["correct"]])
    print(f"\n{P.reader}: 10 seeded answered rows (5 correct, 5 wrong) of {len(ans)} answered / {len(P.rows)} - Read:")
    for r in show:
        print(f"--- {r['trace']} chunk {r['chunk_idx']} {r['pile']} #{r['sample_idx']} | gt {P.trace_obj[r['trace']].gt_answer!r} | extracted {r['answer']!r} | {'CORRECT' if r['correct'] else 'wrong'} | {r['tokens']} tokens{' | REOPENED' if r['reopened'] else ''}")
        print("    ..." + r["completion"][-260:].replace("\n", " ⏎ "))
    boxes = sum("\\boxed" in r["completion"] for r in P.rows)
    result = f"{len(ans)}/{len(P.rows)} answered; {boxes} completions contain \\\\boxed; {len(show)} rows printed for the author (seed {P.cfg['seed']})"
    note(f"P1 extraction {P.reader}", "seeded random rows, answer vs ground truth, human reads", result)


def p1_restarts(P):
    rate = lambda rows: (sum(r["reopened"] for r in rows) / len(rows)) if rows else float("nan")
    by = {}
    for key, t, p in P.pairs:
        for role in ("anchor", "partner"):
            for pile in ("kept", "replaced"):
                by.setdefault((role, pile), []).extend(P.pile(key, p[f"{role}_chunk"], pile))
    per_sentence = sorted(((rate(rows), key, i, pile) for (key, i, pile), rows in P.by.items()), reverse=True)
    heavy = [x for x in per_sentence if x[0] > 0.5]
    top = f"max {per_sentence[0][0]:.2f} at {per_sentence[0][1].split('/')[1]} chunk {per_sentence[0][2]} {per_sentence[0][3]}" if per_sentence else "no piles"
    result = ("re-open rate " + ", ".join(f"{role} {pile} {rate(v):.3f}" for (role, pile), v in sorted(by.items()))
              + f"; {len(heavy)}/{len(per_sentence)} piles above 50% ({top}); "
              + P.readout("d with re-opened rollouts included", reopened_ok=True))
    print(f"{P.reader} restarts: {result}")
    note(f"P1 restarts {P.reader}", "re-open rates from rows.jsonl; d recomputed with re-opened rollouts kept in", result)


def p1_filter(P):
    draw = rng(P.cfg["seed"], "09_filter", P.reader)
    rep = [r for r in P.rows if r["pile"] == "replaced" and not r["reopened"] and r["first_sentence"]]
    pick = lambda pool, k=5: draw.sample(pool, min(k, len(pool)))
    show = pick([r for r in rep if r["dissimilar"]]) + pick([r for r in rep if not r["dissimilar"]])
    print(f"\n{P.reader}: 10 seeded replaced rows (5 kept by the filter, 5 dropped) - Read:")
    for r in sorted(show, key=lambda r: r["similarity"]):
        print(f"--- [{'KEEP' if r['dissimilar'] else 'DROP'} cosine {r['similarity']:.2f}] {r['trace']} chunk {r['chunk_idx']}\n    A: {P.trace_obj[r['trace']].chunks[r['chunk_idx']][:200]!r}\n    B: {r['first_sentence'][:200]!r}")
    surv = [sum(bool(r.get("dissimilar")) and not r["reopened"] for r in rows) for (key, i, pile), rows in P.by.items() if pile == "replaced"]
    zero = [(key.split("/")[1], i) for (key, i, pile), rows in P.by.items() if pile == "replaced" and not any(r.get("dissimilar") and not r["reopened"] for r in rows)]
    result = (f"survivors per replaced pile: median {_med(surv)}, min {min(surv) if surv else 'n/a'}, {sum(x < 30 for x in surv)}/{len(surv)} below 30 after top-ups"
              + (f", {len(zero)} with NO survivor ({', '.join(f'{p} chunk {i}' for p, i in zero)}) - their pairs have no d" if zero else "") + "; "
              + P.readout("d with the filter off", filter_off=True) + f"; {len(show)} rows printed for the author")
    print(f"{P.reader} filter: {result}")
    note(f"P1 filter {P.reader}", "seeded replaced rows with cosine, human reads; survivors; d with the filter off", result)


def p1_kept(P):
    draw = rng(P.cfg["seed"], "09_kept", P.reader)
    pool = [r for r in P.rows if r["pile"] == "kept" and not r["reopened"]]
    show = draw.sample(pool, min(10, len(pool)))
    print(f"\n{P.reader}: 10 seeded kept rows - A's prefilled sentence, then how the reader continues - Read:")
    for r in show:
        print(f"--- {r['trace']} chunk {r['chunk_idx']} #{r['sample_idx']} ({'correct' if r['correct'] else 'wrong'})\n    A: {P.trace_obj[r['trace']].chunks[r['chunk_idx']][:200]!r}\n    B continues: {r['completion'][:240]!r}")
    result = f"{len(show)} kept rows printed for the author (does the reader continue A's sentence rather than restart?)"
    note(f"P1 kept {P.reader}", "seeded kept rows, human reads", result)


def p1_position(P):
    from scipy.stats import spearmanr
    recs = []
    for key, t, p in P.pairs:
        d = P.d_of(key, p)
        for role in ("anchor", "partner"):
            i = p[f"{role}_chunk"]
            kept, rep = P.usable(P.pile(key, i, "kept")), P.usable(P.pile(key, i, "replaced"))
            acc = lambda rows: (lambda a: (sum(bool(r["correct"]) for r in a) / len(a)) if a else None)([r for r in rows if r["answer"]])
            k, r_ = acc(kept), acc(rep)
            recs.append({"reader": P.reader, "trace": key, "role": role, "chunk_idx": i, "rel_position": i / t["n_chunks"],
                         "importance_B": None if k is None or r_ is None else r_ - k, "d": d if role == "anchor" else None,
                         "direction": p["anchor_direction"], "split": t["split"]})
    anchors = [r for r in recs if r["role"] == "anchor" and r["d"] is not None]
    rho, pv = spearmanr([r["rel_position"] for r in anchors], [r["d"] for r in anchors])
    result = f"Spearman(anchor relative position, d) rho {rho:+.2f} (two-sided p {pv:.2f}, n {len(anchors)}); anchor positions median {_med([r['rel_position'] for r in anchors]):.2f} of the trace"
    print(f"{P.reader} position: {result}")
    note(f"P1 position {P.reader}", "d vs relative anchor position; importance_B vs position", result)


# fig10_attrition removed 2026-09-11 at a request; the attrition NUMBERS stay in doc/checks.md.

def p1_caps(P):
    rate = lambda rows, f: (sum(f(r) for r in rows) / len(rows)) if rows else float("nan")
    kept = [r for r in P.rows if r["pile"] == "kept"]; rep = [r for r in P.rows if r["pile"] == "replaced"]
    gaps = []
    for key, t, p in P.pairs:
        for role in ("anchor", "partner"):
            i = p[f"{role}_chunk"]
            pk, pr = P.usable(P.pile(key, i, "kept")), P.usable(P.pile(key, i, "replaced"))
            if pk and pr:
                gaps.append(abs(rate(pk, lambda r: bool(r["answer"])) - rate(pr, lambda r: bool(r["answer"]))))
    result = (f"capped: kept {rate(kept, lambda r: r['capped']):.3f}, replaced {rate(rep, lambda r: r['capped']):.3f}; no answer: kept {rate(kept, lambda r: not r['answer']):.3f}, "
              f"replaced {rate(rep, lambda r: not r['answer']):.3f}; abs p_answered gap between a sentence's piles: median {_med(gaps):.3f}, max {max(gaps) if gaps else float('nan'):.3f}, {sum(g > 0.2 for g in gaps)}/{len(gaps)} above 0.2; "
              + P.readout("d (primary)") + " (the no-answer-counted-wrong readout is 08's)")
    print(f"{P.reader} caps: {result}")
    note(f"P1 caps {P.reader}", "cap / no-answer rates per pile, p_answered gaps", result)


def p1_per_trace(P):
    from scipy.stats import binomtest
    per = {}
    for key, t, p in P.pairs:
        d = P.d_of(key, p)
        if d is not None:
            per.setdefault(key, []).append(d)
    means = {k: sum(v) / len(v) for k, v in per.items()}
    pos = sum(m > 0 for m in means.values())
    bt = binomtest(pos, len(means), 0.5, alternative="greater").pvalue
    result = (f"{pos}/{len(means)} traces with mean d > 0 (sign test p {bt:.3f}); per trace: "
              + ", ".join(f"{k.split('/')[1]} {m:+.2f} (n {len(per[k])})" for k, m in means.items()))
    print(f"{P.reader} per_trace: {result}")
    note(f"P1 per-trace {P.reader}", "per-trace mean d, sign test over traces", result)


P1_MODES = {"recompute": p1_recompute, "extraction": p1_extraction, "restarts": p1_restarts, "filter": p1_filter,
            "kept": p1_kept, "position": p1_position, "caps": p1_caps, "per_trace": p1_per_trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=None, help="screens: --recompute --run_id screen_<reader>")
    ap.add_argument("--readers", nargs="+", default=None, help="Phase 1 modes: default both readers of data/phase1_traces.json")
    ap.add_argument("--all", action="store_true", help="every Phase 1 mode")
    for k in P1_MODES:
        ap.add_argument(f"--{k}", action="store_true")
    args = ap.parse_args()
    if args.run_id:
        if args.recompute:
            sys.exit(0 if recompute_screen(args.run_id) else 1)
        ap.error("with --run_id choose --recompute (screen)")
    modes = list(P1_MODES) if args.all else [k for k in P1_MODES if getattr(args, k)]
    if not modes:
        ap.error("choose a mode")
    import json
    cfg = load_config(REPO_ROOT / "configs/phase1.json")
    sel = json.load(open(REPO_ROOT / "data/phase1_traces.json"))["readers"]
    ok = True
    known = json.load(open(REPO_ROOT / "data/phase1_traces.json"))["by_reader"]
    for reader in args.readers or [sel["B_far"], sel["B_near"]]:
        assert reader in known, f"{reader} is not a reader in data/phase1_traces.json ({list(known)})"
        P = P1(reader, cfg)
        for k in modes:
            res = P1_MODES[k](P)
            ok = ok and (res is not False)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
