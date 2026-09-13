#!/usr/bin/env python
"""Phase 1 analysis (the pre-registered estimand and gate): reader importances,
paired d, CIs, permutation p, Spearman against A, the gate verdict.

    python scripts/08_analyze_anchors.py [--config configs/phase1.json]
        [--readers gpt-oss-20b qwen3-1.7b] [--run_ids p1_gpt-oss-20b p1_qwen3-1.7b]

Inputs: runs/<run_id>/rows.jsonl per reader (07), data/phase1_traces.json.
Per sentence (anchor or partner): importance_B = P(correct | replaced usable) −
P(correct | kept usable), answered-only (src/phase1/piles.importance). Per pair:
d = sign_A × (importance_B(anchor) − importance_B(partner)) (stats.paired_d).
Per reader: mean d, cluster bootstrap over traces + t-interval, one-sided
within-sentence permutation p (stats.permutation_p), fraction d > 0, per-trace
mean d. Beside it, NOT gate inputs (configs/phase1.json importance notes): the
no-answer-counted-incorrect readout, the readout excluding pairs with a short
pile (after the top-ups, < min_answered_per_pile answered in either pile or
< replaced_min_survivors survivors in the replaced one), the |Δ p_answered|
nuisance and its Spearman with d, and Spearman(importance_A, importance_B)
over the reader's sentences with a seeded permutation p.
Gate (config `gate`): positive iff ≥ min_readers readers have p <
max_permutation_p on the answered denominator. Written literally.
Outputs: data/p1_importance.csv (sentences), data/p1_pairs.csv (pairs),
plots/fig5_p1_paired.png (one panel per reader, d per pair sorted),
(fig5b categories and fig6 scatter were removed 2026-09-11 at a request: the category
breakdown has two n=1 cells and a label-permutation p of 0.77, and the pooled scatter is an algebraic
restatement of mean d)
(+ _data.csv each), doc/p1_results.md. A run id may be a comma-joined list: the rows are
pooled (uniqueness then includes the run id).
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import REPO_ROOT, load_config, read_rows, rng, run_dir
from src.common.stats import cluster_bootstrap, paired_d, permutation_p
from src.phase1.piles import PILES, complete, importance, usable

DATA, PLOTS, DOC = REPO_ROOT / "data", REPO_ROOT / "plots", REPO_ROOT / "doc"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#2b2b2b", "#6b6b66"


def load_pairs(reader, run_id, sel, imp):
    rows = [r for rid in run_id.split(",") for r in read_rows(run_dir(rid, create=False) / "rows.jsonl")]   # comma list = pooled batches
    bad = [r for r in rows if r["reader"] != reader or r["phase"] != "p1" or r["run_id"] not in run_id.split(",")]
    assert not bad, f"runs/{run_id}: {len(bad)} rows are not {reader}/p1 rows of these runs (first: {bad[0]['run_id']} {bad[0]['reader']}/{bad[0]['phase']})"
    keys = [(r["run_id"], r["trace"], r["chunk_idx"], r["pile"], r["sample_idx"]) for r in rows]
    assert len(keys) == len(set(keys)), f"runs/{run_id}: duplicate (run, trace, chunk, pile, sample_idx) rows - two 07 processes on one run_id?"
    by = {}
    for r in rows:
        by.setdefault((r["trace"], r["chunk_idx"], r["pile"]), []).append(r)
    out, missing = [], []
    for t in sel["by_reader"][reader]["traces"]:
        key = f"{t['split']}/{t['problem_id']}"
        for p in t["pairs"]:
            piles = {(role, pile): by.get((key, p[f"{role}_chunk"], pile)) for role in ("anchor", "partner") for pile in PILES}
            if any(v is None or not complete(v, pile, imp) for (_, pile), v in piles.items()):   # absent or unfinished pile
                missing.append(f"{key} {p.get('anchor_set', 'matched')} pair {p['anchor_rank']}")
            else:
                out.append({"trace": key, "split": t["split"], "problem_id": t["problem_id"], "pair": p,
                            "piles": {k: usable(v) for k, v in piles.items()}, "raw": piles})
    if missing:
        sys.exit(f"{reader}: {len(missing)} pairs with an absent or unfinished pile in runs/{run_id} (first: {missing[0]}); finish 07")
    return out, missing, len(rows)


def no_answer_wrong(rows):
    """Robustness denominator: a rollout with no answer counts as answered and wrong."""
    return [{**r, "answer": r["answer"] or "<no answer>", "correct": bool(r["correct"]) if r["answer"] else False} for r in rows]


def sentence_record(reader, pr, role, imp):
    p, k_raw, r_raw = pr["pair"], pr["raw"][(role, "kept")], pr["raw"][(role, "replaced")]
    res = importance(k_raw, r_raw)          # pile_stats counts n/reopened/capped/dissimilar on the raw pile and applies usable() itself
    k, r = res["kept"], res["replaced"]
    pa = lambda st: (st["n_answered"] / st["n_usable"]) if st["n_usable"] else None
    return {"reader": reader, "split": pr["split"], "problem_id": pr["problem_id"], "anchor_rank": p["anchor_rank"], "role": role,
            "chunk_idx": p[f"{role}_chunk"], "direction": p["anchor_direction"], "anchor_set": p.get("anchor_set", "matched"),
            "importance_A": p[f"{role}_cf_importance"], "tags": p[f"{role}_tags"], "text": p[f"{role}_text"],
            "importance_B": res["importance"], "p_answered_kept": pa(k), "p_answered_replaced": pa(r),
            "short": k["n_answered"] < imp["min_answered_per_pile"] or r["n_answered"] < imp["min_answered_per_pile"]
                     or r["n_usable"] < imp["replaced_min_survivors"],       # both the pre-registered design floors, as piles.ensure_pile tops up on both
            **{f"kept_{a}": k[a] for a in ("n", "n_reopened", "n_usable", "n_answered", "n_capped", "p_correct")},
            **{f"replaced_{a}": r[a] for a in ("n", "n_reopened", "n_dissimilar", "n_usable", "n_answered", "n_capped", "p_correct")}}


def pair_record(reader, pr, sa, sp):
    d = paired_d((pr["piles"][("anchor", "kept")], pr["piles"][("anchor", "replaced")]),
                 (pr["piles"][("partner", "kept")], pr["piles"][("partner", "replaced")]), pr["pair"]["anchor_direction"])
    gap = lambda s: None if s["p_answered_kept"] is None or s["p_answered_replaced"] is None else abs(s["p_answered_kept"] - s["p_answered_replaced"])
    return {"reader": reader, "split": pr["split"], "problem_id": pr["problem_id"], "anchor_set": pr["pair"].get("anchor_set", "matched"),
            "anchor_rank": pr["pair"]["anchor_rank"], "anchor_chunk": sa["chunk_idx"], "partner_chunk": sp["chunk_idx"], "direction": pr["pair"]["anchor_direction"],
            "anchor_tags": sa["tags"], "importance_A_anchor": sa["importance_A"], "importance_A_partner": sp["importance_A"],
            "importance_B_anchor": sa["importance_B"], "importance_B_partner": sp["importance_B"], "d": d,
            "short": sa["short"] or sp["short"], "answered_gap_anchor": gap(sa), "answered_gap_partner": gap(sp)}


def readout(pairs, transform=None, n_resamples=1000, seed=0):
    """mean d, CIs and the permutation p for a list of pair dicts (as in load_pairs).
    `seed` should already be salted per reader and readout (io.rng)."""
    f = transform or (lambda rows: rows)
    tuples, ds = {}, {}
    for pr in pairs:
        a = (f(pr["piles"][("anchor", "kept")]), f(pr["piles"][("anchor", "replaced")]))
        q = (f(pr["piles"][("partner", "kept")]), f(pr["piles"][("partner", "replaced")]))
        d = paired_d(a, q, pr["pair"]["anchor_direction"])
        if d is not None:
            tuples.setdefault(pr["trace"], []).append((a, q, pr["pair"]["anchor_direction"]))
            ds.setdefault(pr["trace"], []).append(d)
    if not ds:
        return {"n_pairs": 0}
    boot = cluster_bootstrap(ds, n_resamples, seed)
    perm = permutation_p(tuples, n_resamples, seed)
    return {**boot, "p_one_sided": perm["p_one_sided"], "per_trace_mean_d": {t: sum(v) / len(v) for t, v in ds.items()}}


def spearman(xs, ys, groups, n_resamples, seed):
    """Spearman rho with a one-sided permutation p for rho > 0 under the WITHIN-TRACE
    null: y is shuffled inside each group (trace), never across.
    p is NaN when rho is undefined (constant input)."""
    import math
    import warnings
    from scipy.stats import ConstantInputWarning, spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho, p_two = spearmanr(xs, ys)
        if len(xs) < 3 or math.isnan(rho):
            return {"rho": float("nan"), "p_two_sided": float("nan"), "p_perm_one_sided": float("nan"), "n": len(xs)}
        draw, ge = rng(seed, "spearman_within_trace"), 0
        idx = {}
        for j, g in enumerate(groups):
            idx.setdefault(g, []).append(j)
        for _ in range(n_resamples):
            ys2 = list(ys)
            for js in idx.values():
                vals = [ys[j] for j in js]
                draw.shuffle(vals)
                for j, v in zip(js, vals):
                    ys2[j] = v
            if spearmanr(xs, ys2)[0] >= rho:
                ge += 1
    return {"rho": float(rho), "p_two_sided": float(p_two), "p_perm_one_sided": (ge + 1) / (n_resamples + 1), "n": len(xs)}


def _no_nan(x):
    """NaN -> null so the JSON stays strict."""
    import math
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_no_nan(v) for v in x]
    return None if isinstance(x, float) and math.isnan(x) else x


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _style():
    spec = importlib.util.spec_from_file_location("figs", REPO_ROOT / "scripts/90_make_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m._style()


def fig_paired(panels, name, imp):
    """Figure 5. One panel per reader: every anchor/partner pair's d, sorted
    ascending, with mean d, its interval and the zero line. d = sign_A x
    [imp_B(anchor) − imp_B(partner)], so d > 0 means the reader moved the way A's data
    predicts, more at the anchor than at its control; the unit is accuracy. Colour =
    whether A got that trace right or wrong; hollow = a pile that lost rollouts to the
    answer or similarity filters; squares = the extension's opposite-direction anchors,
    if any. Pairs are not labelled by problem: the unit of the claim is the pair, and
    per-trace behaviour is a separate check. The shaded band is the percentile cluster
    bootstrap over the 10 traces - nominal 95%, measured coverage ~90% at 10 clusters,
    so the t-interval on the trace means is the one to quote."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 3.9), squeeze=False,
                             gridspec_kw={"wspace": 0.22})
    for ax, (reader, pairs_recs, ro) in zip(axes.flat, panels):
        recs = [r for r in pairs_recs if r["d"] is not None]
        ordered = sorted(recs, key=lambda r: r["d"])
        lo, hi = ro["ci_bootstrap"]
        ax.axhspan(lo, hi, color="#d9d9d4", alpha=0.6, lw=0, zorder=1)
        ax.axhline(ro["mean_d"], color=INK, lw=1.2, zorder=2)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":", zorder=1)
        for i, r in enumerate(ordered):
            col = BLUE if r["split"].startswith("correct") else ORANGE
            ax.vlines(i, 0, r["d"], color=col, lw=1.0, alpha=0.55, zorder=3)
            ax.scatter(i, r["d"], s=30, color=col, zorder=4, linewidths=1.2,
                       facecolors="none" if r["short"] else col,
                       marker="s" if r.get("anchor_set") == "opposite" else "o")
        ax.set_xticks([]); ax.set_xlim(-1, len(ordered))
        ax.set_xlabel(f"{len(ordered)} anchor/partner pairs, sorted by d")
        ax.set_title(f"{reader}   ·   mean d = {ro['mean_d']:+.3f}  (p = {ro['p_one_sided']:.3f})   ·   "
                     f"d > 0 in {ro['frac_pairs_positive']:.0%}", loc="left", fontsize=9)
        ax.set_ylabel("d")                                   # each panel has its own scale
        ax.spines[["top", "right"]].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color="#d9d9d4"), plt.Line2D([], [], color=INK, lw=1.2)]
    l = ["cluster-bootstrap interval over the 10 traces", "mean d"]
    for c, lab in ((BLUE, "A's trace correct"), (ORANGE, "A's trace wrong")):
        h.append(plt.Line2D([], [], marker="o", linestyle="none", color=c, markersize=6)); l.append(lab)
    h.append(plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", color=MUTED, markersize=6))
    l.append("hollow = a pile with missing rollouts")
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.075, 0.02), frameon=False, ncol=3,
               fontsize=8.4, handletextpad=0.7, columnspacing=2.2)
    fig.suptitle("Is A's anchor more load-bearing for the reader than its partner sentence?",
                 x=0.075, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(PLOTS / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    write_csv(PLOTS / f"{name}_data.csv",
              [{"reader": reader, **r} for reader, pairs_recs, _ in panels for r in pairs_recs if r["d"] is not None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--readers", nargs="+", default=None)
    ap.add_argument("--run_ids", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    sel = json.load(open(REPO_ROOT / "data/phase1_traces.json"))
    readers = args.readers or [sel["readers"]["B_far"], sel["readers"]["B_near"]]
    run_ids = args.run_ids or [f"p1_{r}" for r in readers]
    assert len(readers) == len(run_ids), "--readers and --run_ids must pair up"
    an, gate, imp = cfg["analysis"], cfg["gate"], cfg["importance"]
    n_res = an["permutation_n"]
    salt = lambda *s: rng(cfg["seed"], "p1_analysis", *s).getrandbits(31)      # one seeded stream per reader and readout (io.rng)

    sentences, pairs_out, paired_panels, results = [], [], [], {}
    for reader, run_id in zip(readers, run_ids):
        pairs, missing, n_rows = load_pairs(reader, run_id, sel, imp)
        s_recs, p_recs = [], []
        for pr in pairs:
            sa, sp_ = (sentence_record(reader, pr, role, imp) for role in ("anchor", "partner"))
            s_recs += [sa, sp_]
            p_recs.append(pair_record(reader, pr, sa, sp_))
        primary = readout(pairs, None, n_res, salt(reader, "primary"))
        defined = [pr for pr, r in zip(pairs, p_recs) if r["d"] is not None]          # the primary's pair set
        robust = readout(defined, no_answer_wrong, n_res, salt(reader, "no_answer_wrong"))
        short = [r["short"] for r in p_recs]
        not_short = primary if not any(short) else readout([pr for pr, s in zip(pairs, short) if not s], None, n_res, salt(reader, "not_short"))
        both = [r for r in s_recs if r["importance_B"] is not None]
        trace_of = lambda r: f"{r['split']}/{r['problem_id']}"
        spear = spearman([r["importance_A"] for r in both], [r["importance_B"] for r in both], [trace_of(r) for r in both], n_res, salt(reader, "spearman"))
        anchors_only = [r for r in both if r["role"] == "anchor"]
        spear_anchors = spearman([r["importance_A"] for r in anchors_only], [r["importance_B"] for r in anchors_only], [trace_of(r) for r in anchors_only], n_res, salt(reader, "spearman_anchors"))
        nuis = [(max(r["answered_gap_anchor"] or 0, r["answered_gap_partner"] or 0), r["d"], trace_of(r)) for r in p_recs if r["d"] is not None]
        nuis_sp = spearman([a for a, _, _ in nuis], [d for _, d, _ in nuis], [t for _, _, t in nuis], n_res, salt(reader, "nuisance")) if len(nuis) > 2 else None
        by_cat, cells = {}, {}
        for r in p_recs:
            if r["d"] is not None:
                by_cat.setdefault((r["anchor_tags"] or "untagged").split("|")[0].strip(), []).append(r["d"])
                kind = "insight" if r["direction"] < 0 else "mistake"             # A needed it / A was better without it
                cells.setdefault(f"{kind} anchor on A-{'correct' if r['split'].startswith('correct') else 'wrong'} trace", []).append(r["d"])
        results[reader] = {"run_id": run_id, "n_rows": n_rows, "n_pairs_complete": len(pairs), "missing_pairs": missing,
                           "n_short_pairs": sum(r["short"] for r in p_recs), "primary_answered": primary,
                           "no_answer_counted_wrong": robust, "excluding_short_pairs": not_short,
                           "spearman_all_sentences": spear, "spearman_anchors": spear_anchors, "nuisance_answered_gap_vs_d": nuis_sp,
                           "mean_d_by_anchor_category": {c: {"mean_d": sum(v) / len(v), "n": len(v)} for c, v in sorted(by_cat.items())},
                           "mean_d_by_anchor_kind_and_outcome": {c: {"mean_d": sum(v) / len(v), "n": len(v), "positive": sum(x > 0 for x in v)} for c, v in sorted(cells.items())},
                           "gate_pass": primary.get("p_one_sided", 1.0) < gate["max_permutation_p"]}
        sentences += s_recs
        pairs_out += p_recs
        if primary.get("n_pairs"):
            paired_panels.append((reader, p_recs, primary))
            results[reader]["figures"] = "plots/fig5_p1_paired.png"
        else:
            results[reader]["figures"] = "none (no pair with d defined)"
            print(f"{reader}: no pair with d defined - figures skipped")

    if paired_panels:
        fig_paired(paired_panels, "fig5_p1_paired", imp)

    n_pass = sum(r["gate_pass"] for r in results.values())
    verdict = n_pass >= gate["min_readers"]
    write_csv(DATA / "p1_importance.csv", sentences)
    write_csv(DATA / "p1_pairs.csv", pairs_out)

    lines = [f"# Phase 1 results (generated by scripts/08_analyze_anchors.py; runs: {', '.join(run_ids)})\n",
             f"Gate (configs/phase1.json): positive iff >= {gate['min_readers']} reader(s) have one-sided within-sentence permutation p < "
             f"{gate['max_permutation_p']} for mean d > 0 on the {gate['primary_denominator']} denominator.\n",
             f"**GATE VERDICT: {'PASSED' if verdict else 'NOT PASSED'}** ({n_pass} of {len(results)} readers below p = {gate['max_permutation_p']})."
             + " Numbers are hypotheses until 09_sanity_checks has run.\n"]
    for reader, r in results.items():
        pr, rb, ns, sp = r["primary_answered"], r["no_answer_counted_wrong"], r["excluding_short_pairs"], r["spearman_all_sentences"]
        fmt = lambda ro: (f"mean d {ro['mean_d']:+.3f}, n {ro['n_pairs']} pairs / {ro['n_traces']} traces, d > 0 in {ro['frac_pairs_positive']:.0%}, "
                          f"bootstrap CI [{ro['ci_bootstrap'][0]:+.3f}, {ro['ci_bootstrap'][1]:+.3f}], t-CI [{ro['ci_t_on_cluster_means'][0]:+.3f}, {ro['ci_t_on_cluster_means'][1]:+.3f}], "
                          f"permutation p {ro['p_one_sided']:.3f}") if ro.get("n_pairs") else "no pair with d defined"
        lines += [f"\n## {reader} (runs/{r['run_id']}, {r['n_rows']} rows, {r['n_pairs_complete']} complete pairs"
                  + (f", {len(r['missing_pairs'])} missing" if r["missing_pairs"] else "") + ")\n",
                  f"- primary (answered-only): {fmt(pr)}  -> gate {'PASS' if r['gate_pass'] else 'FAIL'}",
                  f"- no-answer counted incorrect: {fmt(rb)}",
                  f"- excluding {r['n_short_pairs']} pair(s) with a short pile: {fmt(ns)}",
                  f"- Spearman(importance_A, importance_B): all sentences rho {sp['rho']:+.2f} (p_perm {sp['p_perm_one_sided']:.3f}, two-sided {sp['p_two_sided']:.3f}, n {sp['n']}); "
                  f"anchors only rho {r['spearman_anchors']['rho']:+.2f} (p_perm {r['spearman_anchors']['p_perm_one_sided']:.3f}, n {r['spearman_anchors']['n']})",
                  ("- nuisance: Spearman(|Δ p_answered| max over the pair, d) rho " + f"{r['nuisance_answered_gap_vs_d']['rho']:+.2f} (two-sided p {r['nuisance_answered_gap_vs_d']['p_two_sided']:.2f})") if r["nuisance_answered_gap_vs_d"] else "- nuisance: too few pairs",
                  "- per-trace mean d: " + ", ".join(f"{t.split('/')[1]} {v:+.2f}" for t, v in pr.get("per_trace_mean_d", {}).items()),
                  "- mean d by the anchor's first category tag (the pre-registration, descriptive): " + ", ".join(f"{c} {v['mean_d']:+.2f} (n {v['n']})" for c, v in r["mean_d_by_anchor_category"].items()),
                  "- mean d by anchor kind x trace outcome: " + ", ".join(f"{c} {v['mean_d']:+.2f} ({v['positive']}/{v['n']} > 0)" for c, v in r["mean_d_by_anchor_kind_and_outcome"].items()),
                  f"- figures: {r['figures']}"]
    (DOC / "p1_results.md").write_text("\n".join(lines) + "\n")
    (DATA / "p1_results.json").write_text(json.dumps(_no_nan({"gate_verdict": verdict, "out_tag": "", "run_ids": run_ids, "readers": results}), indent=2, default=str))
    print("\n".join(lines))
    print(f"\nwritten: data/p1_importance.csv ({len(sentences)} sentences), data/p1_pairs.csv ({len(pairs_out)} pairs), data/p1_results.json, doc/p1_results.md")


if __name__ == "__main__":
    main()
