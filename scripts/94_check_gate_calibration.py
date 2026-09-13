#!/usr/bin/env python
"""Gate calibration check (python scripts/94_check_gate_calibration.py).

Calibration check for the new metric + permutation null, against the old
folded |.| metric, under a COMPLETE null (p_kept == p_replaced everywhere) with
the asymmetries the design guarantees: anchors near p=0.5, partners locked near
0.85, and a smaller effective n on the anchor's replaced pile."""
import random
import sys

sys.path.insert(0, '.')
from src.common.stats import cluster_bootstrap, permutation_p, p_correct_answered

RNG = random.Random(7)


def pile(p, n):
    return [{"answer": "x", "correct": RNG.random() < p} for _ in range(n)]


def one_experiment(folded, n_traces=10, pairs_per_trace=3):
    d_by_trace, pairs_by_trace = {}, {}
    for t in range(n_traces):
        ds, prs = [], []
        for _ in range(pairs_per_trace):
            pa = RNG.uniform(0.35, 0.65)          # anchor: undetermined
            pp = RNG.uniform(0.75, 0.95)          # partner: locked
            a = (pile(pa, 40), pile(pa, 30))      # anchor replaced pile smaller
            p = (pile(pp, 40), pile(pp, 55))
            if folded:
                f = lambda kr: abs(p_correct_answered(kr[1])[0] - p_correct_answered(kr[0])[0])
                ds.append(f(a) - f(p))
            else:
                f = lambda kr: p_correct_answered(kr[1])[0] - p_correct_answered(kr[0])[0]
                ds.append(f(a) - f(p))
            prs.append((a, p, 1))
        d_by_trace[f"t{t}"] = ds
        pairs_by_trace[f"t{t}"] = prs
    return d_by_trace, pairs_by_trace


def gate_rate(folded, sims, use_permutation):
    fired = 0
    for i in range(sims):
        d_by_trace, pairs = one_experiment(folded)
        cb = cluster_bootstrap(d_by_trace, n_resamples=400, seed=i)
        if use_permutation:
            pv = permutation_p(pairs, n_resamples=200, seed=i)["p_one_sided"]
            if pv < 0.025:                      # the gate as committed (the first review checkpoint): one condition
                fired += 1
        else:
            if cb["ci_bootstrap"][0] > 0 and cb["frac_pairs_positive"] >= 0.65:
                fired += 1
    return fired / sims


SIMS = 200
print(f"complete null, {SIMS} simulated experiments (10 traces x 3 pairs), gate fire rate:")
print(f"  OLD  folded |.|  + bootstrap CI gate : {gate_rate(True,  SIMS, False):.3f}   (target 0.025)")
print(f"  NEW  signed      + bootstrap CI gate : {gate_rate(False, SIMS, False):.3f}")
print(f"  NEW  signed      + permutation p<0.025 (the gate, one condition) : {gate_rate(False, SIMS, True):.3f}")
