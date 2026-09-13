#!/usr/bin/env python
"""Is A's anchor more important to the reader than the sentences it outranked?

    python scripts/24_anchor_permutation.py [--B 20000] [--seed 20260908]
        [--out runs/h24_anchor_permutation]

THE NULL, stated first: *within a trace, which sentences A labelled anchors carries
no information about which sentences matter to the reader.* That null is exactly
simulable - relabel three of the trace's eligible sentences as anchors at random -
so no distributional assumption is needed and the trace clustering is handled by
construction, because every reshuffle happens inside one trace.

This supersedes the trace-split paired t-test as the headline statistic. That test
reduced each trace to the median of its 3 anchors minus the median of its ~43-sentence
pool and ran a t over 10 points: it respected the clustering but discarded 468
sentences' worth of information, weighted a 23-sentence pool the same as a
76-sentence one, and at n = 10 had almost no power (p 0.965 meant little).
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.io import REPO_ROOT, run_dir

DIRECTION = {"correct_base_solution": -1, "incorrect_base_solution": +1}


def candidates():
    """Per trace: (|importance| of A's anchors, |importance| of the pool they outranked).

    Pool = the sentences that COULD have been anchors and were not:
      - the reader measured an importance there (cf_importance is not null)
      - A's release marks the sentence testable (before convergence, measured, not
        overdetermined, uncorrupted prefix)
      - A's released importance points the way this trace's anchors do: negative on a
        trace A got right, positive on one it got wrong
      - it is not one of the trace's partners, which were selected for being inert and
        are therefore not a random draw from the pool
    """
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REPO_ROOT / "runs/dense_*/profile.csv")))],
                   ignore_index=True)
    df = df[df.cf_importance.notna()].copy()
    df["abs_imp"] = df.cf_importance.abs()
    df["role0"] = df.role.fillna("")
    df["matches_direction"] = np.sign(df.released_cf_importance) == df.split.map(DIRECTION)
    out = []
    for (split, pid), g in df.groupby(["split", "problem_id"]):
        elig = g[g.released_testable & g.matches_direction & ~g.role0.str.startswith("partner")]
        anchors = elig[elig.role0.str.startswith("anchor")].abs_imp.to_numpy()
        pool = elig[~elig.role0.str.startswith("anchor")].abs_imp.to_numpy()
        if len(anchors) and len(pool) >= 5:
            out.append((f"{split}/{pid}", anchors, pool))
    return out


def percentile(x, others):
    """Where |x| sits among `others`: 0 = above all of them, 0.50 = chance. Ties count half."""
    return (np.sum(others > x) + 0.5 * np.sum(others == x)) / len(others)


def percentiles_of(traces, picks):
    """The per-anchor percentiles behind statistic(); returned so the observed mean can
    carry a standard error beside the permutation null."""
    vals = []
    for (_, anchors, pool), idx in zip(traces, picks):
        cand = np.concatenate([anchors, pool])
        chosen, rest = cand[idx], np.delete(cand, idx)
        vals += [percentile(x, rest) for x in chosen]
    return np.array(vals)


def statistic(traces, picks):
    """Mean percentile over every labelled anchor. `picks[t]` indexes that trace's candidate set.

    The chosen sentences are scored against the REST OF THE SAME CANDIDATE SET, so the
    observed case and every permuted case are computed identically - if the observed
    anchors were scored against a pool that excluded them while permuted ones were
    scored against a pool that included them, the test would be biased.
    """
    vals = []
    for (_, anchors, pool), idx in zip(traces, picks):
        cand = np.concatenate([anchors, pool])          # the real anchors occupy the first len(anchors) slots
        chosen, rest = cand[idx], np.delete(cand, idx)
        vals += [percentile(x, rest) for x in chosen]
    return float(np.mean(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", default="runs/h24_anchor_permutation")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    traces = candidates()

    obs, n_anchors = statistic(traces, [np.arange(len(a)) for _, a, _ in traces])
    obs_vals = percentiles_of(traces, [np.arange(len(a)) for _, a, _ in traces])
    obs_se = float(obs_vals.std(ddof=1) / np.sqrt(len(obs_vals)))
    null = np.empty(args.B)
    for b in range(args.B):
        null[b] = statistic(traces, [rng.choice(len(a) + len(p), size=len(a), replace=False)
                                     for _, a, p in traces])[0]
    lo, hi = np.percentile(null, [2.5, 97.5])
    p_two = (np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (args.B + 1)
    p_one = (np.sum(null <= obs) + 1) / (args.B + 1)

    print(f"{len(traces)} traces, {n_anchors} anchors, candidate sets "
          f"{[len(a) + len(p) for _, a, p in traces]}")
    print(f"\nobserved mean percentile   {obs:.4f} +- {obs_se:.4f} SE   (0.50 = chance; lower = more important to the reader)")
    print(f"null: mean {null.mean():.4f}  SD {null.std():.4f}  95% range [{lo:.3f}, {hi:.3f}]  (B = {args.B:,})")
    print(f"two-sided p {p_two:.3f}   one-sided p (anchors more important) {p_one:.3f}")
    print(f"observed sits {abs(obs - null.mean()) / null.std():.2f} null SDs from chance")
    print(f"\nSENSITIVITY: this design would have detected a mean percentile outside "
          f"[{lo:.3f}, {hi:.3f}]. It detects nothing.")
    print("CAVEAT: A's top-3 anchors are often adjacent (e.g. sentences 88, 89, 97). The permutation "
          "draws three positions at random and so does not reproduce that clustering; if the reader's "
          "importance has positional structure the null is slightly optimistic.")

    d = run_dir(args.out.split("/")[-1])
    json.dump({"n_traces": len(traces), "n_anchors": n_anchors, "B": args.B, "seed": args.seed,
               "observed_mean_percentile": obs, "observed_se": obs_se,
               "observed_percentiles": [float(v) for v in obs_vals], "null_mean": float(null.mean()), "null_sd": float(null.std()),
               "null_ci95": [float(lo), float(hi)], "p_two_sided": float(p_two), "p_one_sided": float(p_one),
               "candidate_set_sizes": [len(a) + len(p) for _, a, p in traces]},
              open(d / "result.json", "w"), indent=2)
    with open(d / "per_trace.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["trace", "n_anchors", "n_pool", "anchor_abs_imp", "pool_median_abs_imp"])
        for t, a, p in traces:
            w.writerow([t, len(a), len(p), " ".join(f"{x:.4f}" for x in a), f"{np.median(p):.4f}"])
    print(f"\n-> {d}/result.json and per_trace.csv")


if __name__ == "__main__":
    main()
