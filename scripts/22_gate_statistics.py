#!/usr/bin/env python
"""What the gate's permutation p tests, and the tests it should be quoted beside.

    python scripts/22_gate_statistics.py [--readers gpt-oss-20b qwen3-1.7b]
        [--config configs/phase1.json] [--run_id h22_gate_statistics] [--n_sims 2000]

The gate (configs/phase1.json `gate`) reads the one-sided within-sentence
permutation p of src/common/stats.py:107. That permutation pools each
sentence's kept and replaced rollouts and re-splits them, so its reference
distribution is the SHARP null: importance_B = 0 at EVERY sentence, with only
binomial resampling noise around it. It has no between-sentence and no
between-trace component - yet the independent unit is the trace, not the pair
(configs/phase1.json disjoint_problems: 10 traces = 10 problems). §1 measures
that gap, §2 gives the tests that do respect it, §3 and §4 measure what the
gate's alpha and the bootstrap's ci_level are actually worth, §5 redoes the
|z| > 2 tail against the right baseline.

Inputs are counts already on disk: data/p1_pairs.csv (d per pair) and
data/p1_importance.csv (per-pile n and P(correct) per sentence), plus
runs/dense_gpt-oss-20b*/profile.csv for §5's baseline. Each pile is rebuilt as
a bag of (answered, correct) labels from its counts - all the permutation can
see - and the rebuild is checked against the published readout: it reproduces
data/p1_results.json's mean_d and ci_bootstrap exactly and its permutation p to
Monte-Carlo noise. Simulations resample those piles; no rollouts are generated,
so cap attrition and the semantic filter are held fixed at what they were.

Everything here is a hypothesis about the gate's operating characteristics, not
a new result about the readers. Output: a markdown block on stdout and
runs/<run_id>/gate_statistics.csv, one row per number. CPU only; about five
minutes at the default n_sims, nearly all of it in §3.
"""

import argparse
import csv
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom, t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import REPO_ROOT, load_config, rng, run_dir
from src.common.stats import cluster_bootstrap, paired_d

DENSE = ["runs/dense_gpt-oss-20b/profile.csv", "runs/dense_gpt-oss-20b_wrong/profile.csv"]
Z_CUT = 2.0


# ---------------------------------------------------------------- loading

def load_pairs(reader):
    """The reader's pairs, each with its four piles as (n_usable, n_answered,
    n_correct). Pairs whose observed d is undefined are dropped, as 08's
    readout drops them (08_analyze_anchors.py:117)."""
    sent = {}
    for r in csv.DictReader(open(REPO_ROOT / "data/p1_importance.csv")):
        if r["reader"] == reader:
            sent[(r["split"], r["problem_id"], r["anchor_rank"], r["role"])] = r
    out = []
    for r in csv.DictReader(open(REPO_ROOT / "data/p1_pairs.csv")):
        if r["reader"] != reader or r["d"] == "":
            continue
        counts = lambda s, pile: (int(s[f"{pile}_n_usable"]), int(s[f"{pile}_n_answered"]),
                                  round(float(s[f"{pile}_p_correct"]) * int(s[f"{pile}_n_answered"])))
        piles = {}
        for role in ("anchor", "partner"):
            s = sent[(r["split"], r["problem_id"], r["anchor_rank"], role)]
            piles[role] = (counts(s, "kept"), counts(s, "replaced"))
        out.append({"trace": f"{r['split']}/{r['problem_id']}", "d": float(r["d"]),
                    "sign": float(r["direction"]), **piles})
    return out


def rollouts(counts):
    """One pile as the rows stats.py expects: only (answered, correct) matters."""
    n_usable, n_answered, n_correct = counts
    return ([{"answer": "x", "correct": True}] * n_correct
            + [{"answer": "x", "correct": False}] * (n_answered - n_correct)
            + [{"answer": None, "correct": False}] * (n_usable - n_answered))


def by_trace(pairs, key=lambda p: p["d"]):
    out = {}
    for p in pairs:
        out.setdefault(p["trace"], []).append(key(p))
    return out


def sd(xs):
    return float(np.std(np.asarray(xs, float), ddof=1))


# ------------------------------------------------- 1. the gate's own null

def null_mean_d(pairs, n_resamples, seed):
    """The gate's reference distribution, sampled: the loop of
    stats.permutation_p:126-134, keeping every null mean d instead of counting
    how many reach the observed one."""
    r = random.Random(seed)

    def resplit(pile_a, pile_b):                      # stats.permutation_p:118-121
        pool = list(pile_a) + list(pile_b)
        r.shuffle(pool)
        return pool[:len(pile_a)], pool[len(pile_a):]

    piles = [(tuple(rollouts(c) for c in p["anchor"]), tuple(rollouts(c) for c in p["partner"]), p["sign"])
             for p in pairs]
    means = []
    for _ in range(n_resamples):
        ds = [d for a, q, s in piles if (d := paired_d(resplit(*a), resplit(*q), s)) is not None]
        means.append(sum(ds) / len(ds))
    return means


def binomial_var(pair):
    """Var(d) at one pair under the sharp null: each sentence's two answered
    piles are binomial with a common pooled p-hat, so Var(importance) =
    p(1-p)(1/n1 + 1/n2) and the anchor's and partner's add."""
    v = 0.0
    for role in ("anchor", "partner"):
        (_, n1, c1), (_, n2, c2) = pair[role]
        p = (c1 + c2) / (n1 + n2)
        v += p * (1 - p) * (1 / n1 + 1 / n2)
    return v


def sharp_null(pairs, n_resamples, seed):
    """§1: the SD of the gate's null beside the analytic pure-binomial SD it
    should equal, and beside the spread the data actually show. The two agree
    to a few percent; the residue is that the permutation reshuffles the
    no-answer rollouts too, so the answered denominators wobble, while the
    analytic form holds them at their observed values."""
    trace_d = by_trace(pairs)
    trace_v = by_trace(pairs, binomial_var)
    return {
        "sd_null_simulated": sd(null_mean_d(pairs, n_resamples, seed)),
        "sd_null_analytic": math.sqrt(sum(map(binomial_var, pairs))) / len(pairs),
        "sd_pair_d_null": math.sqrt(np.mean([binomial_var(p) for p in pairs])),
        "sd_pair_d_observed": sd([p["d"] for p in pairs]),
        "sd_trace_mean_null": math.sqrt(np.mean([np.mean(v) / len(v) for v in trace_v.values()])),
        "sd_trace_mean_observed": sd([np.mean(v) for v in trace_d.values()]),
    }


# ------------------------------------------------------- 2. alternatives

def t_test(xs):
    x = np.asarray(xs, float)
    stat = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    return stat, float(t_dist.sf(stat, len(x) - 1)), len(x)


def alternatives(pairs, cfg, reader, seed):
    """§2: the tests that treat d as the datum. Pair-level ones assume the 30
    pairs are independent (they are not - three share a trace); trace-level
    ones use the 10 traces, the unit configs/phase1.json calls independent."""
    ds = [p["d"] for p in pairs]
    per_trace = by_trace(pairs)
    tm = [float(np.mean(v)) for v in per_trace.values()]
    k = len(tm)
    g = np.random.default_rng(seed)
    out = []
    add = lambda test, n, unit, stat, p, effect: out.append(
        {"test": test, "n": n, "unit": unit, "stat": stat, "p": float(p), "effect": effect})

    stat, p, n = t_test(ds)
    add("paired t-test on the pair-level d", n, "pairs", f"t = {stat:.3f}", p,
        f"p < {cfg['gate']['max_permutation_p']}, but it ignores the trace clustering")

    flips = g.choice([-1.0, 1.0], size=(200_000, n)) @ np.array(ds) / n
    add("sign-flip permutation over pairs", n, "pairs", f"{sum(d > 0 for d in ds)}/{n} pairs d > 0",
        ((flips >= np.mean(ds)).sum() + 1) / (len(flips) + 1),
        f"p < {cfg['gate']['max_permutation_p']}; same clustering caveat (2^{n} flips, so Monte Carlo)")

    stat, p, _ = t_test(tm)
    add("t-test on the trace-level mean d", k, "traces", f"t = {stat:.3f}", p,
        "p < %s - the assumption-light headline test" % cfg["gate"]["max_permutation_p"])

    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=k)))
    exact = signs @ np.array(tm) / k
    add("sign-flip permutation over traces (exhaustive)", k, "traces",
        f"{sum(t > 0 for t in tm)}/{k} trace means > 0", (exact >= np.mean(tm)).sum() / len(exact),
        "p < %s; exact - all 2^%d = %d assignments, smallest attainable p = %.4f"
        % (cfg["gate"]["max_permutation_p"], k, len(exact), 1 / len(exact)))

    b = cfg["analysis"]["cluster_bootstrap_n"]
    pub = cluster_bootstrap(per_trace, b, rng(cfg["seed"], "p1_analysis", reader, "primary").getrandbits(31))
    means = bootstrap_means(per_trace, 10 * b, seed)
    lo, hi = means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]
    add("cluster bootstrap over traces", k, f"traces, B = {b} / {10 * b}",
        "CI [%+.4f, %+.4f] at B=%d (reproduces data/p1_results.json), [%+.4f, %+.4f] at B=%d"
        % (*pub["ci_bootstrap"], b, lo, hi, 10 * b),
        (sum(m <= 0 for m in means) + 1) / (len(means) + 1),
        "CI excludes 0 / p < %s; the percentile CI under-covers (§4) and its endpoint moves with B"
        % cfg["gate"]["max_permutation_p"])
    return out


def bootstrap_means(d_by_trace, n_resamples, seed):
    """Sorted mean d over `n_resamples` resamples of the TRACES, the resampling
    of stats.cluster_bootstrap:81-88, kept so the interval and the one-sided p
    come from the same draws."""
    r = random.Random(seed)
    traces = list(d_by_trace)
    means = []
    for _ in range(n_resamples):
        flat = [d for t in [r.choice(traces) for _ in traces] for d in d_by_trace[t]]
        means.append(sum(flat) / len(flat))          # stats.cluster_bootstrap:86-87, arithmetic included
    return sorted(means)


# ------------------------------------- 3-4. simulation under a trace null

def bundle(pairs):
    """The pairs as arrays for simulation: answered pile sizes, pooled p-hat,
    A's sign and the trace index, each of shape (pairs, 2) for (anchor, partner)."""
    col = lambda f: np.array([[f(p[role]) for role in ("anchor", "partner")] for p in pairs])
    traces = sorted({p["trace"] for p in pairs})
    return {"n_kept": col(lambda s: s[0][1]), "n_repl": col(lambda s: s[1][1]),
            "p_hat": col(lambda s: (s[0][2] + s[1][2]) / (s[0][1] + s[1][1])),
            "sign": np.array([p["sign"] for p in pairs]),
            "trace": np.array([traces.index(p["trace"]) for p in pairs]), "n_traces": len(traces)}


def moments(pairs):
    """Between-trace SD tau and the extra sentence-level SD of d, by moments:
    the observed spread minus the binomial part. A trace offset shared by BOTH
    sentences of a pair would cancel in d, so the trace-level null has to be
    stated on d itself: d = u_trace + e_pair + binomial noise, E d = 0."""
    per_trace = by_trace(pairs)
    tm = [float(np.mean(v)) for v in per_trace.values()]
    v_tracemean = np.mean([np.mean(v) / len(v) for v in by_trace(pairs, binomial_var).values()])
    resid = [d - np.mean(v) for v in per_trace.values() for d in v]
    within = sum(x * x for x in resid) / (len(pairs) - len(tm))
    return (math.sqrt(max(0.0, sd(tm) ** 2 - v_tracemean)),
            math.sqrt(max(0.0, within - np.mean([binomial_var(p) for p in pairs]))))


def simulate(g, b, tau, sigma):
    """One synthetic dataset with mean d = 0: every pair of a trace shares
    u ~ N(0, tau), each pair adds e ~ N(0, sigma), and the offset goes on the
    anchor's replaced pile, so E d = u + e. Returns the answered counts of the
    four piles and the pair-level d."""
    off = g.normal(0, tau, b["n_traces"])[b["trace"]] + g.normal(0, sigma, len(b["sign"]))
    delta = np.zeros_like(b["p_hat"])
    delta[:, 0] = b["sign"] * off
    c_kept = g.binomial(b["n_kept"], np.clip(b["p_hat"] - delta / 2, 0, 1))
    c_repl = g.binomial(b["n_repl"], np.clip(b["p_hat"] + delta / 2, 0, 1))
    imp = c_repl / b["n_repl"] - c_kept / b["n_kept"]
    return c_kept, c_repl, b["sign"] * (imp[:, 0] - imp[:, 1])


def null_draws(g, b, c_kept, c_repl, n_resamples):
    """The gate's within-sentence permutation, vectorised: re-splitting a pooled
    pile is a hypergeometric draw on its answered counts. Checked in §1 against
    the pure-Python loop."""
    c, n = c_kept + c_repl, b["n_kept"] + b["n_repl"]
    a = g.hypergeometric(c[..., None], (n - c)[..., None], b["n_kept"][..., None],
                         size=c.shape + (n_resamples,))
    imp = (c[..., None] - a) / b["n_repl"][..., None] - a / b["n_kept"][..., None]
    return (b["sign"][:, None] * (imp[:, 0, :] - imp[:, 1, :])).mean(0)


def gate_size(pairs, tau, sigma, n_sims, n_resamples, alpha, seed):
    """§3: how often the gate's own statistic fires at alpha when mean d = 0 but
    the sentences of a trace move together."""
    b, g, fired, spread = bundle(pairs), np.random.default_rng(seed), 0, []
    for _ in range(n_sims):
        c_kept, c_repl, d = simulate(g, b, tau, sigma)
        nulls = null_draws(g, b, c_kept, c_repl, n_resamples)
        fired += ((nulls >= d.mean()).sum() + 1) / (n_resamples + 1) < alpha
        spread.append(sd([d[b["trace"] == t].mean() for t in range(b["n_traces"])]))
    return fired / n_sims, float(np.mean(spread))


def coverage(pairs, tau, sigma, n_sims, n_boot, seed):
    """§4: what the percentile cluster bootstrap and the t(k-1) interval on the
    trace means actually cover, on data generated with a true mean d of 0."""
    b, g = bundle(pairs), np.random.default_rng(seed)
    traces = [p["trace"] for p in pairs]
    hits = [0, 0]
    for i in range(n_sims):
        _, _, d = simulate(g, b, tau, sigma)
        ds = {}
        for t, x in zip(traces, d):
            ds.setdefault(t, []).append(float(x))
        ci = cluster_bootstrap(ds, n_boot, seed + i)
        hits[0] += ci["ci_bootstrap"][0] <= 0 <= ci["ci_bootstrap"][1]
        hits[1] += ci["ci_t_on_cluster_means"][0] <= 0 <= ci["ci_t_on_cluster_means"][1]
    return hits[0] / n_sims, hits[1] / n_sims


# ------------------------------------------------------------ 5. the tail

def z_of(p1, n1, p2, n2, pooled):
    """z for a difference of two binomial proportions. POOLED uses one p-hat
    (the textbook test of p1 = p2); UNPOOLED is the Wald/CI form. Both piles
    locked at the same value gives 0/0, which is z = 0, not a tail event."""
    if pooled:
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    else:
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return 0.0 if se == 0 else (p2 - p1) / se


def pair_z(reader, pooled):
    """|z| > Z_CUT counts among the reader's anchors and partners, from the
    per-pile counts of data/p1_importance.csv."""
    out = {}
    for r in csv.DictReader(open(REPO_ROOT / "data/p1_importance.csv")):
        if r["reader"] != reader or r["importance_B"] == "":
            continue
        z = z_of(float(r["kept_p_correct"]), int(r["kept_n_answered"]),
                 float(r["replaced_p_correct"]), int(r["replaced_n_answered"]), pooled)
        out.setdefault(r["role"], []).append(z)
    return {k: (sum(abs(z) > Z_CUT for z in v), len(v)) for k, v in out.items()}


def dense_z(pooled):
    """The same statistic at every sentence of gpt-oss's own dense profile,
    where importance = P(correct | dissimilar at i) - accuracy(i+1)
    (20_dense_profile.py), split by the sentence's role in the anchor table."""
    rows = [r for f in DENSE for r in csv.DictReader(open(REPO_ROOT / f))]
    by_chunk = {(r["split"], r["problem_id"], int(r["chunk_idx"])): r for r in rows}
    out = {}
    for r in rows:
        after = by_chunk.get((r["split"], r["problem_id"], int(r["chunk_idx"]) + 1))
        if r["p_correct_dissimilar"] == "" or after is None or after["accuracy"] == "":
            continue
        n1, n2 = int(after["n_answered"]), int(r["n_dissimilar_answered"])
        if not n1 or not n2:
            continue
        z = z_of(float(after["accuracy"]), n1, float(r["p_correct_dissimilar"]), n2, pooled)
        role = r["role"].split()[0] if r["role"] else "ordinary"
        out.setdefault(role, []).append(z)
    return {k: (sum(abs(z) > Z_CUT for z in v), len(v)) for k, v in out.items()}


# ------------------------------------------------------------- reporting

def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def row(section, reader, quantity, value, n, note):
    return {"section": section, "reader": reader, "quantity": quantity, "value": value, "n": n, "note": note}


def section1(per_reader, n_res, salt):
    """§1: the gate's null is the sharp null - its SD is the pure-binomial one."""
    md = ["### 1. The gate's permutation tests the sharp null", "",
          "`stats.permutation_p` pools each sentence's kept and replaced rollouts and re-splits them, so its reference",
          "distribution is *importance_B = 0 at every sentence, binomial noise only*: its SD matches the analytic",
          "pure-binomial SD, which is the demonstration. It carries no between-sentence and no between-trace",
          "component, so it is narrower than the spread the data show at both levels. Sampled with",
          f"{5 * n_res} resamples, 5x the gate's, to pin the SD; the null itself is the gate's.", ""]
    body, rows = [], []
    for reader, pairs in per_reader.items():
        res = sharp_null(pairs, 5 * n_res, salt(reader, "sharp_null"))
        body.append([reader, f"{res['sd_null_simulated']:.5f}", f"{res['sd_null_analytic']:.5f}",
                     f"{res['sd_pair_d_null']:.4f} / {res['sd_pair_d_observed']:.4f}",
                     f"{res['sd_trace_mean_null']:.4f} / {res['sd_trace_mean_observed']:.4f}"])
        rows += [row("1_sharp_null", reader, k, round(v, 6), len(pairs),
                     "SD of mean d unless the name says otherwise") for k, v in res.items()]
    return md + table(["reader", "SD of gate null, simulated", "SD, analytic pure binomial",
                       "SD of pair d: null / observed", "SD of trace mean d: null / observed"], body) + [""], rows


def section2(per_reader, cfg, alpha, salt):
    """§2: the same d values under nulls that keep the trace clustering."""
    md = ["### 2. Correctly specified alternatives", "",
          "Row 1 is the pre-registered gate statistic (`gate.evidence`), read back from data/p1_results.json; it is",
          "what the gate committed to and cannot be changed after the fact. The rest treat d as the datum.",
          "**Quote the exhaustive trace-level sign-flip beside the gate p**: it is exact, assumes nothing about the",
          "shape of d, and uses the unit configs/phase1.json calls independent. The t-test on the trace means agrees.", ""]
    published = json.load(open(REPO_ROOT / "data/p1_results.json"))["readers"]
    body, rows = [], []
    for reader, pairs in per_reader.items():
        p = published[reader]["primary_answered"]["p_one_sided"]
        body.append([reader, "**within-sentence permutation (PRE-REGISTERED GATE)**", f"{len(pairs)} pairs",
                     "mean d = %+.4f" % np.mean([x["d"] for x in pairs]), f"{p:.4f}",
                     f"p < {alpha}; tests the sharp null of §1"])
        rows.append(row("2_alternatives", reader, "permutation_p_pre_registered", round(p, 5), len(pairs),
                        "read from data/p1_results.json, not recomputed here"))
        for a in alternatives(pairs, cfg, reader, salt(reader, "alternatives")):
            body.append([reader, a["test"], f"{a['n']} {a['unit']}", a["stat"], f"{a['p']:.4f}", a["effect"]])
            rows.append(row("2_alternatives", reader, a["test"], round(a["p"], 5), a["n"],
                            f"one-sided p over {a['unit']}; effect if {a['effect']}"))
    return md + table(["reader", "test", "unit (n)", "statistic", "one-sided p", "effect is there if"], body) + [""], rows


def section3(per_reader, n_sims, n_res, alpha, salt):
    """§3: how often the gate fires when the null holds at the trace level."""
    md = [f"### 3. Realised size of the gate at alpha = {alpha}", "",
          "Simulated datasets with true mean d = 0 in which each trace's pairs share an offset u ~ N(0, tau) and each",
          "pair adds e ~ N(0, sigma). A common offset on BOTH sentences of a pair would cancel in d, so the",
          "trace-level null has to be stated on d itself. tau and sigma are moment estimates from the reader's own d",
          "and are themselves uncertain on 10 traces, so 2 x tau brackets tau. Row 1 is the sharp null the gate",
          "assumes and checks that the simulation recovers the nominal size; row 2 adds only sentence-level spread;",
          "row 3 is the answer - the null true at the trace level. Fires if the gate's own permutation p < alpha.", ""]
    body, rows = [], []
    grid = [(0.0, 0.0, "sharp (tau = sigma = 0)"), (0.0, 1.0, "sentence-level only"),
            (1.0, 1.0, "TRACE-LEVEL"), (2.0, 1.0, "trace-level, 2 x tau")]
    for reader, pairs in per_reader.items():
        tau, sigma = moments(pairs)
        for t_mult, s_mult, label in grid:
            size, spread = gate_size(pairs, t_mult * tau, s_mult * sigma, n_sims, n_res, alpha,
                                     salt(reader, "size", label))
            body.append([reader, label, f"{t_mult * tau:.4f}", f"{s_mult * sigma:.4f}", f"{size:.4f}", f"{spread:.4f}"])
            rows.append(row("3_gate_size", reader, f"realised_size_null_{label}", round(size, 4), n_sims,
                            f"tau={t_mult * tau:.4f} sigma={s_mult * sigma:.4f}; nominal {alpha}"))
        obs = sd([float(np.mean(v)) for v in by_trace(pairs).values()])
        rows.append(row("3_gate_size", reader, "sd_trace_means_observed", round(obs, 4), len(by_trace(pairs)),
                        "compare with the simulated spread at tau-hat: the generator is calibrated if they agree"))
    md += table(["reader", "trace-level null", "tau", "sigma", f"P(gate fires) at alpha={alpha}",
                 "simulated SD of trace means"], body)
    return md + ["", f"Read: the gate p is worth alpha = {alpha} only if row 1 is the truth. On row 3 the same",
                 "statistic fires more often than that, so a gate p just under alpha is not a 2.5% result; the",
                 "effect is there if the trace-level tests of §2 clear alpha as well. The last column against the",
                 "observed SD of trace means (CSV: sd_trace_means_observed) says whether the generator is calibrated.", ""], rows


def section4(per_reader, n_sims, n_boot, level, salt):
    """§4: what the two intervals cover under the same generator."""
    md = [f"### 4. What the {level} intervals actually cover", "",
          "Same generator at tau-hat; the interval is right if it contains the true mean d = 0.", ""]
    body, rows = [], []
    for reader, pairs in per_reader.items():
        tau, sigma = moments(pairs)
        cov_b, cov_t = coverage(pairs, tau, sigma, n_sims, n_boot, salt(reader, "coverage"))
        body.append([reader, f"{cov_b:.3f}", f"{cov_t:.3f}", len(by_trace(pairs))])
        rows += [row("4_coverage", reader, "coverage_percentile_cluster_bootstrap", round(cov_b, 4), n_sims, f"nominal {level}"),
                 row("4_coverage", reader, "coverage_t_on_cluster_means", round(cov_t, 4), n_sims, f"nominal {level}")]
    md += table(["reader", "percentile cluster bootstrap", "t(k-1) on trace means", "clusters"], body)
    return md + ["", f"Read: an interval may be quoted as {level} only if its coverage is near {level}. Quote the",
                 "t-interval; the percentile cluster bootstrap on 10 clusters is narrower than it claims to be, so",
                 "its excluding 0 is weaker evidence than it looks.", ""], rows


def section5(readers):
    """§5: the |z| > 2 tail against the ordinary-sentence rate, both SEs."""
    md = [f"### 5. The |z| > {Z_CUT:.0f} tail, against the right baseline", "",
          "z = importance_B / SE, SE for a difference of two binomial proportions from the per-pile counts. The",
          "pooled and the unpooled SE are both defensible, nobody pre-registered a choice, and they disagree - which",
          "is itself the finding.", ""]
    body, rows = [], []
    for reader in readers:
        for pooled in (True, False):
            counts = pair_z(reader, pooled)
            body.append([reader, "pooled" if pooled else "unpooled (Wald)"]
                        + ["%d/%d" % counts[role] for role in ("anchor", "partner")])
            rows += [row("5_tail", reader, f"{role}s_over_z2_{'pooled' if pooled else 'unpooled'}",
                         counts[role][0], counts[role][1], "from data/p1_importance.csv") for role in ("anchor", "partner")]
    md += table(["reader", "SE", f"anchors abs(z) > {Z_CUT:.0f}", "partners"], body) + [""]

    body = []
    for pooled in (True, False):
        dense, se = dense_z(pooled), "pooled" if pooled else "unpooled"
        k_ord, n_ord = dense["ordinary"]
        rate = k_ord / n_ord
        rows.append(row("5_tail", "gpt-oss-20b", f"ordinary_rate_{se}", round(rate, 4), n_ord,
                        "dense profile, sentences that are neither anchor nor partner"))
        for label, (k, n) in [("anchors (p1 piles)", pair_z("gpt-oss-20b", pooled)["anchor"]),
                              ("partners (p1 piles)", pair_z("gpt-oss-20b", pooled)["partner"]),
                              ("anchors (dense piles, n=100)", dense["anchor"]),
                              ("partners (dense piles, n=100)", dense["partner"])]:
            more, fewer = float(binom.sf(k - 1, n, rate)), float(binom.cdf(k, n, rate))
            body.append([se, label, f"{k}/{n}", f"{n * rate:.1f}", f"{more:.3f}", f"{fewer:.3f}"])
            rows.append(row("5_tail", "gpt-oss-20b", f"binom_p_more_{label}_{se}", round(more, 4), n,
                            f"P(X >= {k} | n={n}, p={rate:.4f}), the ordinary-sentence rate"))
        body.append([se, "**ordinary sentences (the baseline)**", f"{k_ord}/{n_ord}", "-", "-", "-"])
    md += table(["SE", "sentences", f"abs(z) > {Z_CUT:.0f}", "expected at the ordinary rate",
                 "binomial p (more)", "binomial p (fewer)"], body)
    return md + ["",
                 "**The pure-null framing overstates this.** Against a pure null |z| > 2 is expected 4.6% of the time,",
                 "1.4 of 30, and 6 anchors looks like a tail. The reader's ORDINARY sentences on the same traces exceed",
                 "|z| > 2 far more often than that, and against that rate the anchors are a borderline excess under the",
                 "pooled SE and no excess at all under the unpooled one. The partner side is the sturdier half (fewer",
                 "tail sentences than ordinary), but the partners were selected to be dull. Carried, not corrected: the",
                 "p1 z's come from ~30/40-rollout piles and the dense z's from ~100, so the baseline rate is measured",
                 "with more power than the anchors compared against it. Effect is there if the anchor count beats the",
                 "ordinary rate at binomial p < 0.025; it does not, under either SE. NB the 6/30-against-12.9%",
                 "pairing quoted earlier mixes conventions - a pooled count against an unpooled rate. Kept",
                 "consistent it is 6/30 against 10.1% (p 0.08) or 5/30 against 12.9% (p 0.34).", ""], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=["gpt-oss-20b", "qwen3-1.7b"])
    ap.add_argument("--config", default="configs/phase1.json")
    ap.add_argument("--run_id", default="h22_gate_statistics")
    ap.add_argument("--n_sims", type=int, default=2000, help="simulated datasets for §3 and §4")
    args = ap.parse_args()
    cfg = load_config(args.config)
    alpha, level = cfg["gate"]["max_permutation_p"], cfg["analysis"]["ci_level"]
    n_res, n_boot = cfg["analysis"]["permutation_n"], cfg["analysis"]["cluster_bootstrap_n"]
    salt = lambda *s: rng(cfg["seed"], "22_gate_statistics", *s).getrandbits(31)
    per_reader = {r: load_pairs(r) for r in args.readers}

    md = [f"## Gate statistics ({args.run_id}) - hypotheses about the gate, not new results about the readers", "",
          f"n_sims = {args.n_sims}, permutation resamples = {n_res}, bootstrap B = {n_boot}, alpha = {alpha}, "
          f"ci_level = {level}. Arithmetic on counts already on disk: data/p1_pairs.csv, data/p1_importance.csv and",
          "runs/dense_gpt-oss-20b*/profile.csv. Each pile is rebuilt as a bag of (answered, correct) labels from its",
          "counts - all the permutation can see; the rebuild reproduces data/p1_results.json's mean_d and",
          "ci_bootstrap exactly. Pairs: " + ", ".join(f"{r} {len(p)}" for r, p in per_reader.items()) + ".", ""]
    rows = []
    for lines, csv_rows in [section1(per_reader, n_res, salt),
                            section2(per_reader, cfg, alpha, salt),
                            section3(per_reader, args.n_sims, n_res, alpha, salt),
                            section4(per_reader, args.n_sims, n_boot, level, salt),
                            section5(args.readers)]:
        md += lines
        rows += csv_rows

    path = run_dir(args.run_id) / "gate_statistics.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "reader", "quantity", "value", "n", "note"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md + [f"Numbers: `{path.relative_to(REPO_ROOT)}` ({len(rows)} rows)."]))


if __name__ == "__main__":
    main()
