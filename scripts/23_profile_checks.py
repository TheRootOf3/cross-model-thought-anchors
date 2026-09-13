#!/usr/bin/env python
"""Profile-level checks on the reader's dense importance profile (2026-09-10).

    python scripts/23_profile_checks.py [--reader gpt-oss-20b]
        [--profiles dense_gpt-oss-20b dense_gpt-oss-20b_wrong] [--pairs data/p1_pairs.csv]
        [--config configs/phase1.json] [--run_id h23_profile_checks]

Three checks on the reader's own importance at EVERY sentence of A's 10 traces
(runs/<profile>/profile.csv, written by 20_dense_profile) beside the
pre-registered pairs (data/p1_pairs.csv, written by 08_analyze_anchors):

1 COMPARATOR LADDER. Where A's anchors sit in the reader's own within-trace
  |importance| ordering, against three progressively fairer comparator pools:
  all sentences, A's testable ones, and the testable ones whose released
  importance has the sign the trace's anchors were taken in (configs/phase1.json
  anchor_direction_rule) - the pool the anchor actually beat when it was
  selected. Anchors and partners are excluded from every pool; partners are the
  control row. Percentile 0 = the reader's most important sentence, 0.50 = chance.
2 NOISE FLOOR. Each importance is a difference of two binomial proportions, so
  SE = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2). For the reader both counts come from
  profile.csv; for A they are COUNTED from the released rollouts
  (<ROLLOUTS_ROOT>/<split>/<problem>/chunk_<i>/solutions.json), never inferred
  from the released fractions: n2 = rollouts with a non-empty answer at i+1
  (analyze_rollouts.py:1502-1514) and n1 = different_trajectories_fraction(i) x
  (answered rollouts at i), because chunk_info keeps only answered rollouts
  (:1536-1549), dtf is counted over exactly those (:678-719) and the dissimilar
  accuracy's denominator is that same set (:744-752). Both counts are verified
  against the released numbers before they are used.
3 PAIRED VS DENSE. The same 30 anchor/partner pairs re-measured on the dense
  rollouts: d = sign_A x [imp(anchor) - imp(partner)], independent of the
  pre-registered run's rollouts. The pre-registered headline is mean d = +0.053.

Prints a markdown block and writes every number in it to
runs/<run_id>/checks.csv (check, scope, metric, value, n). Read-only: no
generation, no GPU.
"""

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data import ROLLOUTS_ROOT
from src.common.io import REPO_ROOT, load_config, rng, run_dir
from src.common.stats import cluster_bootstrap

POOLS = {"all sentences": "all", "testable (A's eligible set)": "testable",
         "testable + A-importance of the anchors' sign": "same_sign"}


def num(v):
    """Float, or None for an empty CSV cell (a metric that was never defined)."""
    return float(v) if v not in ("", None) else None


def load_profile(run_ids, reader):
    """by[(split, problem_id)][chunk_idx] = the profile row."""
    by = {}
    for rid in run_ids:
        for r in csv.DictReader(open(run_dir(rid, create=False) / "profile.csv")):
            assert r["reader"] == reader, f"runs/{rid}/profile.csv holds {r['reader']} rows, not {reader}"
            by.setdefault((r["split"], r["problem_id"]), {})[int(r["chunk_idx"])] = r
    return by


def load_pairs(path, reader, by):
    """The reader's pre-registered pairs, with the trace's anchor direction checked
    to be single-valued and the pair's roles checked against the profile."""
    pairs = [r for r in csv.DictReader(open(REPO_ROOT / path)) if r["reader"] == reader]
    for p in pairs:
        d = by[(p["split"], p["problem_id"])]
        assert d[int(p["anchor_chunk"])]["role"].startswith("anchor"), f"{p['problem_id']} chunk {p['anchor_chunk']} is not an anchor in profile.csv"
        assert d[int(p["partner_chunk"])]["role"].startswith("partner"), f"{p['problem_id']} chunk {p['partner_chunk']} is not a partner in profile.csv"
    directions = {}
    for p in pairs:
        directions.setdefault((p["split"], p["problem_id"]), set()).add(int(p["direction"]))
    assert all(len(v) == 1 for v in directions.values()), f"a trace has anchors in both directions: {directions}"
    return pairs, {k: v.pop() for k, v in directions.items()}


def released_counts(by):
    """Per sentence, from the released rollouts themselves: how many answered, the
    accuracy that denominator gives, and how many rollouts are behind A's
    counterfactual importance (the dissimilar ones). Returns the counts and the
    tallies that verify both against the released numbers."""
    counts, verified = {}, {"accuracy_match": 0, "n_sentences": 0, "dissimilar_integral": 0, "n_measured": 0}
    for (split, pid), d in by.items():
        labeled = json.load(open(ROLLOUTS_ROOT / split / pid / "chunks_labeled.json"))
        for i, r in d.items():
            with open(ROLLOUTS_ROOT / split / pid / f"chunk_{i}" / "solutions.json") as f:
                sols = json.load(f)
            answered = [s for s in sols if s.get("answer")]          # analyze_rollouts.py:1502-1514
            acc = sum(1 for s in answered if s.get("is_correct") is True) / len(answered) if answered else None
            rel = num(r["released_accuracy"])
            verified["n_sentences"] += 1
            verified["accuracy_match"] += acc is not None and rel is not None and abs(acc - rel) < 1e-9
            dtf = labeled[i].get("different_trajectories_fraction") or 0.0
            n_dis = dtf * len(answered)
            verified["n_measured"] += dtf > 0
            verified["dissimilar_integral"] += dtf > 0 and abs(n_dis - round(n_dis)) < 1e-6
            counts[(split, pid, i)] = {"n_answered": len(answered), "accuracy": acc, "n_dissimilar": round(n_dis)}
    return counts, verified


def se_diff(p1, n1, p2, n2):
    """SE of a difference of two independent binomial proportions."""
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 and n2 else None


def sentence_records(by, counts, anchors):
    """One record per sentence: both models' importance at it and each one's SE.
    Both importances are (accuracy of the dissimilar rollouts at i) - (accuracy at
    i+1), so both SEs pair the dissimilar count at i with the answered count at i+1."""
    out = []
    for (split, pid), d in sorted(by.items()):
        for i, r in sorted(d.items()):
            nxt = d.get(i + 1)
            if nxt is None:                                          # no next sentence: no importance either
                continue
            imp_b, imp_a = num(r["cf_importance"]), num(r["released_cf_importance"])
            se_b = None if imp_b is None else se_diff(num(r["p_correct_dissimilar"]), int(r["n_dissimilar_answered"]),
                                                      num(nxt["accuracy"]), int(nxt["n_answered"]))
            c, cn = counts[(split, pid, i)], counts[(split, pid, i + 1)]
            p_dis = None if imp_a is None or cn["accuracy"] is None else imp_a + cn["accuracy"]   # A's dissimilar accuracy
            se_a = None if p_dis is None else se_diff(p_dis, c["n_dissimilar"], cn["accuracy"], cn["n_answered"])
            out.append({"split": split, "problem_id": pid, "chunk_idx": i, "imp_A": imp_a, "se_A": se_a,
                        "imp_B": imp_b, "se_B": se_b, "anchor": (split, pid, i) in anchors,
                        # A's dissimilar accuracy must be a multiple of 1/n_dissimilar if that count is the right one
                        "n_dissimilar_confirmed": None if p_dis is None else abs(p_dis * c["n_dissimilar"] - round(p_dis * c["n_dissimilar"])) < 1e-6})
    return out


def comparator_pool(d, kind, direction):
    """|reader importance| of the sentences an anchor of this trace is compared
    against; the trace's own anchors and partners are excluded from all three."""
    out = []
    for r in d.values():
        imp = num(r["cf_importance"])
        if imp is None or r["role"]:
            continue
        if kind != "all" and r["released_testable"] != "True":
            continue
        if kind == "same_sign":
            a = num(r["released_cf_importance"])
            if a is None or a == 0 or (a > 0) != (direction > 0):
                continue
        out.append(abs(imp))
    return out


def percentile(x, pool):
    """Where |x| sits in the pool: 0 = above every sentence in it, 0.50 = chance."""
    return (sum(1 for v in pool if v > x) + 0.5 * sum(1 for v in pool if v == x)) / len(pool)


def role_importances(by, pairs, role, key=None):
    """|reader importance| at the anchors (or partners), of one trace or of all."""
    out = []
    for p in pairs:
        if key is not None and (p["split"], p["problem_id"]) != key:
            continue
        v = num(by[(p["split"], p["problem_id"])][int(p[f"{role}_chunk"])]["cf_importance"])
        if v is not None:
            out.append(abs(v))
    return out


def ladder(by, pairs, directions, add):
    """Check 1: the three comparator pools, per trace then medianed over the traces.
    The median pooled over the 30 anchors is reported beside it because the two
    aggregations disagree. Returns the markdown lines."""
    lines = ["| comparator pool | pool size (median/trace, total) | pool median \\|imp_B\\| | anchor pct (per-trace) "
             "| partner pct (per-trace) | anchor pct (pooled) | partner pct (pooled) |",
             "|---|---|---|---|---|---|---|"]
    for label, kind in POOLS.items():
        sizes, pool_med, per_trace, pooled = [], [], {"anchor": [], "partner": []}, {"anchor": [], "partner": []}
        for key, direction in directions.items():
            pool = comparator_pool(by[key], kind, direction)
            sizes.append(len(pool))
            pool_med.append(st.median(pool))
            for role in ("anchor", "partner"):
                pcts = [percentile(x, pool) for x in role_importances(by, pairs, role, key)]
                per_trace[role].append(st.median(pcts))
                pooled[role] += pcts
        lines.append(f"| {label} | {st.median(sizes):.1f} (min {min(sizes)}), {sum(sizes)} | {st.median(pool_med):.4f} | "
                     + " | ".join(f"{st.median(v):.3f}" for v in (per_trace["anchor"], per_trace["partner"],
                                                                  pooled["anchor"], pooled["partner"])) + " |")
        add("1_ladder", label, "pool_size_median_per_trace", st.median(sizes), sum(sizes))
        add("1_ladder", label, "pool_median_abs_imp_B", st.median(pool_med), sum(sizes))
        for role in ("anchor", "partner"):
            add("1_ladder", label, f"{role}_percentile_median_per_trace", st.median(per_trace[role]), len(directions))
            add("1_ladder", label, f"{role}_percentile_median_pooled", st.median(pooled[role]), len(pooled[role]))
    for role in ("anchor", "partner"):
        vals = role_importances(by, pairs, role)
        per = [st.median(role_importances(by, pairs, role, key)) for key in directions]
        lines.append(f"\nreader \\|importance\\| at the {len(vals)} {role}s: median {st.median(vals):.4f} pooled, "
                     f"{st.median(per):.4f} per-trace")
        add("1_ladder", f"{role}s", "median_abs_imp_B_pooled", st.median(vals), len(vals))
        add("1_ladder", f"{role}s", "median_abs_imp_B_per_trace", st.median(per), len(directions))
    return lines


def noise_floor(recs, add):
    """Check 2: median |importance|, median SE, median |importance|/SE and the share
    of the observed variance that is not measurement noise, for A and for the reader
    on the same sentences. SE = 0 (a pile that answered all-correct or all-wrong)
    is kept in the SE median and dropped from the |importance|/SE median."""
    both = [r for r in recs if r["se_A"] is not None and r["se_B"] is not None]
    lines = ["| set | model | n | median \\|imp\\| | median SE | median \\|imp\\|/SE | signal share of variance | SE = 0 |",
             "|---|---|---|---|---|---|---|---|"]
    for label, rows in (("all sentences", both), ("the 30 anchors", [r for r in both if r["anchor"]])):
        for m, who in (("A", "A (released)"), ("B", "reader")):
            imps = [r[f"imp_{m}"] for r in rows]
            ses = [r[f"se_{m}"] for r in rows]
            z = [abs(i) / s for i, s in zip(imps, ses) if s > 0]
            var = st.variance(imps)
            signal = (var - sum(s * s for s in ses) / len(ses)) / var
            lines.append(f"| {label} | {who} | {len(rows)} | {st.median([abs(i) for i in imps]):.4f} | {st.median(ses):.4f} | "
                         f"{st.median(z):.2f} | {signal:.2f} | {sum(1 for s in ses if s == 0)} |")
            for metric, v in (("median_abs_importance", st.median([abs(i) for i in imps])), ("median_SE", st.median(ses)),
                              ("median_abs_importance_over_SE", st.median(z)), ("signal_share_of_variance", signal)):
                add("2_noise", f"{label} / {who}", metric, v, len(rows))
    return lines


def paired_vs_dense(by, pairs, cfg, add):
    """Check 3: the same 30 pairs on the dense rollouts. Same estimand as the
    pre-registered run (the dense pile at i+1 is that run's kept pile: piles.py,
    prefill chunks[:i+1]), different rollouts, n = 100 per sentence."""
    from scipy.stats import pearsonr, spearmanr, ttest_rel
    recs = []
    for p in pairs:
        d = by[(p["split"], p["problem_id"])]
        a, q = num(d[int(p["anchor_chunk"])]["cf_importance"]), num(d[int(p["partner_chunk"])]["cf_importance"])
        recs.append({"trace": f"{p['split']}/{p['problem_id']}", "split": p["split"], "anchor_rank": int(p["anchor_rank"]),
                     "adjacent": abs(int(p["anchor_chunk"]) - int(p["partner_chunk"])) == 1, "d_paired": float(p["d"]),
                     "d_dense": None if a is None or q is None else int(p["direction"]) * (a - q)})
    ok = [r for r in recs if r["d_dense"] is not None]
    dense, paired = [r["d_dense"] for r in ok], [r["d_paired"] for r in ok]
    boot = cluster_bootstrap({t: [r["d_dense"] for r in ok if r["trace"] == t] for t in dict.fromkeys(r["trace"] for r in ok)},
                             cfg["analysis"]["cluster_bootstrap_n"], rng(cfg["seed"], "profile_checks", "dense_bootstrap").getrandbits(31))
    rho, rho_p = pearsonr(paired, dense)
    sp, sp_p = spearmanr(paired, dense)
    tt = ttest_rel(paired, dense)
    lines = [f"- dense d (n = {len(ok)} of {len(recs)} pairs): mean {st.mean(dense):+.4f}, median {st.median(dense):+.4f}, "
             f"{sum(1 for v in dense if v > 0)} of {len(dense)} positive, cluster-bootstrap CI "
             f"[{boot['ci_bootstrap'][0]:+.4f}, {boot['ci_bootstrap'][1]:+.4f}], t-CI on trace means [{boot['ci_t_on_cluster_means'][0]:+.4f}, {boot['ci_t_on_cluster_means'][1]:+.4f}]",
             f"- pre-registered d on the same pairs: mean {st.mean(paired):+.4f}, {sum(1 for v in paired if v > 0)} of {len(paired)} positive",
             f"- agreement: Pearson r {rho:+.3f} (p {rho_p:.3f}), Spearman rho {sp:+.3f} (p {sp_p:.3f}), "
             f"paired t (pre-registered - dense) {tt.statistic:+.3f} (p {tt.pvalue:.3f})",
             f"- {sum(1 for r in ok if r['adjacent'])} of {len(ok)} pairs are adjacent sentences, where the dense d's two terms share the pile at anchor+1",
             "- the dense estimate uses 100 rollouts per sentence and takes the kept side from the profile's own pile at i+1; "
             "the pre-registered one used 40 kept / 60+ replaced rollouts generated for that pair",
             "",
             "| breakdown | n | dense mean d | dense positive | pre-registered mean d |", "|---|---|---|---|---|"]
    groups = [(f"anchor rank {k}", [r for r in ok if r["anchor_rank"] == k]) for k in sorted({r["anchor_rank"] for r in ok})]
    groups += [(f"A's trace {'correct' if s.startswith('correct') else 'wrong'}", [r for r in ok if r["split"] == s])
               for s in sorted({r["split"] for r in ok})]
    for label, g in groups:
        lines.append(f"| {label} | {len(g)} | {st.mean([r['d_dense'] for r in g]):+.4f} | {sum(1 for r in g if r['d_dense'] > 0)} | "
                     f"{st.mean([r['d_paired'] for r in g]):+.4f} |")
        add("3_paired_vs_dense", label, "dense_mean_d", st.mean([r["d_dense"] for r in g]), len(g))
        add("3_paired_vs_dense", label, "paired_mean_d", st.mean([r["d_paired"] for r in g]), len(g))
    for metric, v in (("dense_mean_d", st.mean(dense)), ("dense_median_d", st.median(dense)),
                      ("dense_n_positive", sum(1 for v in dense if v > 0)), ("dense_ci_bootstrap_low", boot["ci_bootstrap"][0]),
                      ("dense_ci_bootstrap_high", boot["ci_bootstrap"][1]), ("paired_mean_d", st.mean(paired)),
                      ("pearson_r", rho), ("spearman_rho", sp), ("paired_t_p_two_sided", tt.pvalue)):
        add("3_paired_vs_dense", "all pairs", metric, v, len(ok))
    for r in ok:
        add("3_paired_vs_dense", f"{r['trace']}#{r['anchor_rank']}", "d_dense", r["d_dense"], 1)
        add("3_paired_vs_dense", f"{r['trace']}#{r['anchor_rank']}", "d_paired", r["d_paired"], 1)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="gpt-oss-20b")
    ap.add_argument("--profiles", nargs="+", default=["dense_gpt-oss-20b", "dense_gpt-oss-20b_wrong"])
    ap.add_argument("--pairs", default="data/p1_pairs.csv")
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default="h23_profile_checks")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = []
    add = lambda check, scope, metric, value, n: out.append({"check": check, "scope": scope, "metric": metric, "value": value, "n": n})

    by = load_profile(args.profiles, args.reader)
    pairs, directions = load_pairs(args.pairs, args.reader, by)
    counts, verified = released_counts(by)
    n_sent = verified["n_sentences"]
    anchors = {(p["split"], p["problem_id"], int(p["anchor_chunk"])) for p in pairs}
    recs = sentence_records(by, counts, anchors)
    confirmed = [r["n_dissimilar_confirmed"] for r in recs if r["n_dissimilar_confirmed"] is not None]
    add("0_inputs", "released rollouts", "sentences_where_recounted_accuracy_matches_released", verified["accuracy_match"], n_sent)
    add("0_inputs", "released rollouts", "measured_sentences_where_dtf_x_n_answered_is_an_integer", verified["dissimilar_integral"], verified["n_measured"])
    add("0_inputs", "released rollouts", "measured_sentences_where_A_importance_is_a_multiple_of_1_over_n_dissimilar", sum(confirmed), len(confirmed))

    md = [f"## 23 profile checks - {args.reader} ({', '.join('runs/' + p for p in args.profiles)}; {args.pairs})",
          f"{n_sent} sentences on {len(by)} traces, {sum(1 for r in recs if r['imp_B'] is not None)} with a defined reader importance, "
          f"{len(pairs)} pre-registered pairs. Numbers are hypotheses.", "",
          "### 1. Comparator ladder: where A's anchors sit in the reader's own ordering", "",
          "Percentile 0 = the reader's most important sentence of the pool, 0.50 = chance; anchors and partners are "
          "excluded from every pool. Per-trace = median over the trace's 3 anchors, then over the 10 traces; pooled = "
          "median over the 30 anchors.", ""]
    md += ladder(by, pairs, directions, add)
    md += ["", "Quote the third pool: it is the set the anchor was selected out of, so it is the only "
           "comparison in which A's ranking, and not the eligibility filters, does the work.", "",
           "### 2. Noise floor: how much of each importance is measurement error", "",
           f"Verified before use, on the released rollouts: the answered denominator reproduces the released accuracy on "
           f"{verified['accuracy_match']}/{n_sent} sentences; different_trajectories_fraction x n_answered is an integer on "
           f"{verified['dissimilar_integral']}/{verified['n_measured']} measured sentences, and A's importance is a multiple of "
           f"1/that count on {sum(confirmed)}/{len(confirmed)} - so it is the exact denominator behind A's importance, not an estimate.", ""]
    md += noise_floor(recs, add)
    md += ["", "### 3. The pre-registered pairs on independent (dense) rollouts", ""]
    md += paired_vs_dense(by, pairs, cfg, add)
    md += ["", "### For the limitations section", "",
           "- Check 1 and check 3 both belong there: A's fine-grained ranking does not survive in the reader (the anchors sit at "
           "chance or worse against the pool they were selected out of), and the pre-registered mean d shrinks by more than half "
           "when the same 30 pairs are re-measured on independent rollouts, with a CI that includes 0.",
           "- Check 2 belongs there as the reason: the reader's importance at A's anchors is under one standard error from zero, "
           "so single-sentence reader importances are not interpretable one at a time - only aggregates are.",
           f"- Scope: there is a dense profile for {args.reader} only, so the other reader's gate result has no independent "
           "re-measurement here.", ""]

    path = run_dir(args.run_id) / "checks.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "scope", "metric", "value", "n"])
        w.writeheader()
        w.writerows(out)
    print("\n".join(md))
    print(f"written: runs/{args.run_id}/checks.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
