"""Phase 1 statistics: paired d, its permutation null, and the cluster bootstrap.

the pre-registered estimand. Two decisions of 2026-09-08 live here:

SIGNED importance, not |.|. thought-anchors' own metric is a signed difference,
`diff = dissimilar_accuracy - next_accuracy` (analyze_rollouts.py:773), with
abs() applied only under --absolute, which defaults False (:36). We use the
same quantity and the same sign order (replaced - kept), so importance_B is
commensurable with the released importance_A and the pre-registered design Spearman compares like
with like. It also removes a bias that a folded |.| introduces: under the null
E|D| = 0.798*s > 0, so d = importance(anchor) - importance(partner) was biased
upward whenever the anchor's pile was noisier - which the design guarantees,
since anchors sit where p is near 0.5 and partners where it is locked, and the
answered-only denominator makes the effective n differ too. Simulated, the
folded gate fired on 20-68% of complete nulls instead of 2.5%.

ORIENTED by A. Anchors are selected by |importance_A| (since 2026-09-09 in the
direction matching the trace outcome: negative on correct traces, positive on
wrong ones - config anchor_direction_rule; "either" is the direction-free
rule), so each pair is
oriented by the sign of A's released importance at the anchor:
d = sign_A * (importance_B(anchor) - importance_B(partner)). Positive d = B's
outcome moves with the anchor the way A's did. The sign comes from A's data,
fixed before any reader rollout exists, so the null of d stays centred on 0
and nothing is folded. decided 2026-09-08 (second review checkpoint).

PERMUTATION NULL for the gate. Under H0 the kept and replaced rollouts of one
sentence are exchangeable, so shuffling their labels within the sentence and
recomputing gives the null distribution of mean d directly - absorbing any
asymmetry in n, in p, in cap attrition or in filter survival, without
assuming a form for it. The cluster bootstrap stays for the interval, but note
its percentile CI covers ~89%, not 95%, on 10 clusters; report the t-interval
on the cluster means beside it.
"""

import math
import random


def p_correct_answered(rollouts: list[dict]) -> tuple[float | None, int]:
    """(P(correct), n_answered) with no-answer rollouts dropped from numerator
    and denominator, as every thought-anchors accuracy does. None when nothing
    was answered - the caller must not turn that into 0.0."""
    answered = [r for r in rollouts if r.get("answer")]
    if not answered:
        return None, 0
    return sum(bool(r["correct"]) for r in answered) / len(answered), len(answered)


def importance_b(kept: list[dict], replaced: list[dict]) -> float | None:
    """Signed reader importance of one sentence, in the source's sign order:
    P(correct | replaced) - P(correct | kept). Negative = the sentence was
    load-bearing (replacing it hurts). None if either pile answered nothing."""
    pk, nk = p_correct_answered(kept)
    pr, nr = p_correct_answered(replaced)
    if pk is None or pr is None:
        return None
    return pr - pk


def paired_d(anchor: tuple, partner: tuple, direction: float = 1.0) -> float | None:
    """d for one anchor/partner pair, oriented by A: `direction` is the sign
    (+1/-1) of A's released importance at the anchor. Each pile argument is
    (kept, replaced)."""
    ia, ip = importance_b(*anchor), importance_b(*partner)
    return None if ia is None or ip is None else direction * (ia - ip)


def _mean(xs):
    return sum(xs) / len(xs)


def cluster_bootstrap(d_by_trace: dict[str, list[float]], n_resamples: int = 1000,
                      seed: int = 0) -> dict:
    """Percentile CI for mean d, resampling TRACES (the independent unit -
    configs/phase1.json disjoint_problems makes the 10 traces 10 problems).

    Also returns the t-interval on the per-trace means: with 10 clusters the
    percentile CI covers ~89% while t(9) is nominal, so report both and let the
    gate read the permutation p-value rather than either interval alone."""
    traces = list(d_by_trace)
    flat = [d for ds in d_by_trace.values() for d in ds]
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        pick = [rng.choice(traces) for _ in traces]
        means.append(_mean([d for t in pick for d in d_by_trace[t]]))
    means.sort()
    lo, hi = means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]

    per_trace = [_mean(ds) for ds in d_by_trace.values() if ds]
    k = len(per_trace)
    m = _mean(per_trace)
    if k > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in per_trace) / (k - 1))
        from scipy.stats import t as t_dist
        t = float(t_dist.ppf(0.975, k - 1))     # df = k-1 (a hand table here was off by one row until 2026-09-09)
        half = t * sd / math.sqrt(k)
        t_lo, t_hi = m - half, m + half
    else:
        t_lo = t_hi = float("nan")
    return {"mean_d": _mean(flat), "n_pairs": len(flat), "n_traces": k,
            "ci_bootstrap": (lo, hi), "ci_t_on_cluster_means": (t_lo, t_hi),
            "frac_pairs_positive": sum(d > 0 for d in flat) / len(flat)}


def permutation_p(pairs_by_trace: dict[str, list[tuple]], n_resamples: int = 1000,
                  seed: int = 0) -> dict:
    """One-sided p for mean d > 0 under the null that, within each sentence,
    the kept and replaced rollouts are exchangeable.

    `pairs_by_trace[trace]` is a list of (anchor, partner, direction); anchor
    and partner are (kept_rollouts, replaced_rollouts), direction is A's sign
    at the anchor. Each resample re-splits every sentence's pooled rollouts
    into piles of the original sizes and recomputes mean d."""
    rng = random.Random(seed)

    def shuffled(pile_a, pile_b):
        pool = list(pile_a) + list(pile_b)
        rng.shuffle(pool)
        return pool[:len(pile_a)], pool[len(pile_a):]

    observed = _mean([d for ds in pairs_by_trace.values()
                      for a, p, s in ds if (d := paired_d(a, p, s)) is not None])
    ge = 0
    for _ in range(n_resamples):
        ds = []
        for pairs in pairs_by_trace.values():
            for (a_kept, a_rep), (p_kept, p_rep), s in pairs:
                d = paired_d(shuffled(a_kept, a_rep), shuffled(p_kept, p_rep), s)
                if d is not None:
                    ds.append(d)
        if ds and _mean(ds) >= observed:
            ge += 1
    return {"mean_d": observed, "p_one_sided": (ge + 1) / (n_resamples + 1),
            "n_resamples": n_resamples}
