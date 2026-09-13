#!/usr/bin/env python
"""Lead figure: where do A's anchors sit in the READER's own importance ordering?

    python scripts/98_fig_anchor_rank.py

Reads runs/dense_<reader>*/profile.csv (the reader's counterfactual importance at
every sentence of its 10 traces, ~100 rollouts each) and writes one pooled figure
per reader. 90_make_figures.py's fig11 asks the same question but emits one panel
per trace outcome, so the pooled n = 30 statistics had to be made by hand; this
script computes them.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.io import REPO_ROOT


def _profiles_or_exit():
    """The dense profiles these figures are drawn from, or a message saying how to make them."""
    paths = sorted((REPO_ROOT / "runs").glob("dense_*/profile.csv"))
    if not paths:
        raise SystemExit("no runs/dense_*/profile.csv found. These figures need the dense profile: run "
                         "scripts/20_dense_profile.py (GPU), or unpack the data bundle's rollouts/dense_* "
                         "directories into runs/ - see the README.")
    return paths


def _figs():
    """scripts/90_make_figures.py as a module: _style, _finish and the colours."""
    spec = importlib.util.spec_from_file_location("figs", REPO_ROOT / "scripts/90_make_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def percentiles(pool: pd.DataFrame) -> dict[int, float]:
    """chunk_idx -> its rank / n in the pool, sorted by DESCENDING |reader
    importance|: 1/n is the reader's most important sentence, ~0.50 is chance.
    mergesort on a chunk-ordered frame breaks ties towards the lower index, as
    02_build_anchor_table.assign does (pandas' default sort is unstable: three
    sentences tied at 0.0100 once moved a percentile from 0.839 to 0.847)."""
    order = pool.cf_importance.abs().sort_values(ascending=False, kind="mergesort").index
    return {int(c): (k + 1) / len(pool) for k, c in enumerate(pool.chunk_idx[order])}


def selection_pool(g: pd.DataFrame) -> pd.DataFrame:
    """The sentences A's anchors were picked from in this trace: testable, and
    released importance of the sign matching the trace outcome (negative on a
    correct trace, positive on a wrong one) - 02_build_anchor_table.assign,
    rule match_trace_outcome."""
    want = -1 if g.split.iloc[0].startswith("correct") else 1
    return g[g.released_testable & (g.released_cf_importance * want > 0)]


TOP3_MIN_DISSIMILAR = 20      # a top-3 diamond must rest on >= 20 surviving resamples; without
                              # it, two sentences of problem 330 top the trace on a single rollout


def strip_panel(ax, figs, df, traces, rows=None, note=True):
    """The per-trace strip of the reader's |counterfactual importance|: every measured
    sentence (grey, square-root axis), the trace median, the reader's own top 3 (among
    sentences whose replaced pile kept >= TOP3_MIN_DISSIMILAR rollouts) and A's 3 anchors.
    Used by figure 12a; `rows`, if given, collects what is drawn for the _data.csv."""
    import numpy as np
    n_sent = 0
    for x, key in enumerate(traces):
        g = df[(df.split == key[0]) & (df.problem_id == key[1]) & df.cf_importance.notna()]
        mags = g.cf_importance.abs()
        ok = g[g.n_dissimilar_answered >= TOP3_MIN_DISSIMILAR]
        top3 = list(ok.cf_importance.abs().sort_values(ascending=False, kind="mergesort").index[:3])
        role = g.role.fillna("")            # written by 20_dense_profile.py:125-128 from data/phase1_traces.json
        ax.scatter([x] * len(g), mags, s=7, color=figs.ALL, alpha=0.55, lw=0, zorder=2,
                   label="every measured sentence" if x == 0 else None)
        ax.plot([x - 0.3, x + 0.3], [mags.median()] * 2, color=figs.MUTED, lw=1.1, zorder=3,
                label="trace median" if x == 0 else None)
        ax.scatter([x] * 3, mags[top3], s=38, marker="D", facecolors="none", zorder=4,
                   edgecolors=figs.PARTNER, lw=1.2, label="the reader's own top 3" if x == 0 else None)
        anchors = g[role.str.startswith("anchor")]
        ax.scatter([x] * len(anchors), anchors.cf_importance.abs(), s=32, color=figs.ANCHOR, zorder=5,
                   edgecolors=figs.SURFACE, lw=0.5, label="A's 3 anchors" if x == 0 else None)
        n_sent += len(g)
        if rows is not None:
            rows += [(g.reader.iloc[0], key[0], key[1], int(r.chunk_idx), round(abs(r.cf_importance), 4),
                      role[i], i in top3, int(r.n_dissimilar_answered), round(float(mags.median()), 4), len(g))
                     for i, r in g.iterrows()]
    n_ok = sum(k[0].startswith("correct") for k in traces)      # the correct-trace strips come first
    ax.axvspan(n_ok - 0.5, len(traces) - 0.4, color="#f1f1ed", lw=0, zorder=0)
    ax.text((n_ok - 1) / 2, 1.02, "A's trace was correct", ha="center", va="bottom", fontsize=8.5, color=figs.MUTED)
    ax.text((n_ok + len(traces) - 1) / 2, 1.02, "A's trace was wrong", ha="center", va="bottom", fontsize=8.5, color=figs.MUTED)
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.set_yticks([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(traces)))
    ax.set_xticklabels([k[1].split("_")[1] for k in traces], fontsize=8)
    ax.set_xlim(-0.6, len(traces) - 0.4)
    ax.set_xlabel("MATH problem id")
    short = ax.get_figure().get_size_inches()[1] < 3.5          # a squeezed panel cannot fit one line
    ax.set_ylabel("|counterfactual importance|\n\u221a scale" if short
                  else "|counterfactual importance|, \u221a scale", fontsize=8.5 if short else None)
    if note:
        ax.text(1.0, -0.17, f"{n_sent} measured sentences over {len(traces)} traces", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=figs.MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=figs.GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    return n_sent


def fig_anchor_strip():
    """Figure 12a. Where A's anchors sit in the reader's own importance profile.

    Each strip is one of A's 10 released MATH traces, read by gpt-oss-20b only. Grey: every
    sentence of that trace at which the reader's counterfactual importance could be measured
    - |P(correct | the sentence is resampled to something dissimilar, cosine < 0.8 to the
    original) − P(correct | the trace continues from the next sentence)|, 100 rollouts per
    sentence, answered rollouts only, square-root axis. 1,436 of the 1,459 sentences are
    plotted: 10 are each trace's last sentence, which has no next-sentence baseline, and 13
    had no resample survive the similarity filter. Bar: the trace's median. Blue: A's three
    anchors. Orange diamonds: the three sentences the READER ranks highest, among those whose
    replaced pile kept at least 20 rollouts - without that floor two sentences of problem 330
    top their trace on a single surviving rollout, at 0.98 and 0.83.

    The statistic printed above is the anchors' mean percentile within their own trace's
    candidate set (runs/h24_anchor_permutation; 0.50 = chance, lower = more important to the
    reader), with the standard error over the 30 anchors and the two-sided p of a within-trace
    label permutation, B = 20,000. Restricting the ranking to piles of >= 20 surviving
    resamples moves it to 0.520, so the null is not an artefact of thin piles."""
    figs = _figs()
    figs._style()
    import json
    import matplotlib.pyplot as plt
    h24_path = REPO_ROOT / "runs/h24_anchor_permutation/result.json"
    if not h24_path.exists():
        raise SystemExit(f"missing {h24_path.relative_to(REPO_ROOT)} - run scripts/24_anchor_permutation.py first "
                         f"(it needs the dense profiles in runs/dense_*/profile.csv).")
    h24 = json.load(open(h24_path))
    prof = pd.concat([pd.read_csv(p) for p in _profiles_or_exit()],
                     ignore_index=True)
    for reader, df in prof.groupby("reader", sort=False):
        traces = list(dict.fromkeys(zip(df.split, df.problem_id)))
        fig, ax = plt.subplots(figsize=(9.4, 2.7))
        rows = []
        strip_panel(ax, figs, df, traces, rows)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.30), frameon=False, ncol=4, fontsize=8.5)
        fig.suptitle(f"{reader}: A's anchors are not where the reader's own anchors are",
                     x=0.02, y=1.17, ha="left", fontsize=11, fontweight="bold")
        fig.text(0.02, 1.07,
                 f"A's {h24['n_anchors']} anchors: mean within-trace percentile "
                 f"{h24['observed_mean_percentile']:.2f} \u00b1 {h24['observed_se']:.2f} SE against 0.50 at chance",
                 fontsize=8.2, color=figs.MUTED, va="top")
        figs._finish(fig, f"fig12a_anchor_strip_{reader}", rows,
                     ["reader", "split", "problem_id", "chunk_idx", "abs_importance_B", "role", "reader_top3",
                      "n_dissimilar_answered", "trace_median_abs_importance_B", "n_sentences_measured"])


def fig_percentile_curves():
    """Figure 12b. Every trace's importance curve, with A's anchors on it.

    One faint line per trace: the reader's |counterfactual importance| at each sentence of
    that trace's candidate set, sorted from the sentence that matters most to the reader
    (percentile 0) to the one that matters least (percentile 1), on a square-root axis. The
    candidate set is the one A's anchors were drawn from - measured, testable in A's release,
    and A's importance pointing the way the trace's outcome does - 20 to 76 sentences per
    trace (median 43), so a trace's curve and the anchors on it are directly comparable.
    Blue dots: where A's three anchors land on their own trace's curve. The curves show that
    traces differ a great deal in how much any sentence matters to the reader - problem 2137
    runs an order of magnitude above problem 4019 at the same percentile - which is why the
    test is a within-trace percentile rather than a pooled comparison of importances. The
    dots scatter across the whole width: mean percentile 0.53 (dashed line) against 0.50 at
    chance, permutation p = 0.58 (runs/h24_anchor_permutation)."""
    figs = _figs()
    figs._style()
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    h24_path = REPO_ROOT / "runs/h24_anchor_permutation/result.json"
    if not h24_path.exists():
        raise SystemExit(f"missing {h24_path.relative_to(REPO_ROOT)} - run scripts/24_anchor_permutation.py first "
                         f"(it needs the dense profiles in runs/dense_*/profile.csv).")
    h24 = json.load(open(h24_path))
    prof = pd.concat([pd.read_csv(p) for p in _profiles_or_exit()],
                     ignore_index=True)
    for reader, df in prof.groupby("reader", sort=False):
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        rows = []
        for (split, pid), g in df.groupby(["split", "problem_id"], sort=False):
            g = g[g.cf_importance.notna()]
            pool = selection_pool(g)
            if len(pool) < 5:
                continue
            role = pool.role.fillna("")
            v = pool.assign(abs_imp=pool.cf_importance.abs(), role0=role)
            v = v.sort_values("abs_imp", ascending=False, kind="mergesort")
            pct = np.arange(len(v)) / (len(v) - 1)
            correct = split.startswith("correct")
            ax.plot(pct, v.abs_imp, color=figs.MUTED, lw=0.9, alpha=0.45, zorder=2)
            anc = [(x, y, int(c)) for x, y, r, c in zip(pct, v.abs_imp, v.role0, v.chunk_idx)
                   if r.startswith("anchor")]
            ax.scatter([x for x, _, _ in anc], [y for _, y, _ in anc], s=34, zorder=4,
                       color=figs.ANCHOR if correct else figs.PARTNER, edgecolors=figs.SURFACE, lw=0.6,
                       label=None)
            rows += [(reader, split, pid, int(c), round(float(y), 4), round(float(x), 4),
                      r if isinstance(r, str) else "", len(v))
                     for x, y, r, c in zip(pct, v.abs_imp, v.role0, v.chunk_idx)]
        ax.axvline(h24["observed_mean_percentile"], color=figs.INK, lw=1.1, ls=(0, (4, 3)), zorder=3)
        ax.axvline(0.5, color=figs.MUTED, lw=0.8, ls=":", zorder=1)
        ax.set_yscale("function", functions=(np.sqrt, np.square))
        ax.set_yticks([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]); ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("percentile within the trace's candidate set\n(0 = the sentence that matters most to the reader)")
        ax.set_ylabel("|counterfactual importance|, \u221a scale")
        ax.grid(axis="y", color=figs.GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        h = [plt.Line2D([], [], color=figs.MUTED, lw=1.2, alpha=0.6),
             plt.Line2D([], [], color=figs.ANCHOR, marker="o", ls="", ms=6),
             plt.Line2D([], [], color=figs.PARTNER, marker="o", ls="", ms=6),
             plt.Line2D([], [], color=figs.INK, lw=1.1, ls=(0, (4, 3)))]
        ax.legend(h, ["one trace (its candidate sentences, sorted)", "A's anchor, trace A got right",
                      "A's anchor, trace A got wrong",
                      f"mean anchor percentile {h24['observed_mean_percentile']:.2f} (chance 0.50)"],
                  loc="upper left", bbox_to_anchor=(0, -0.28), frameon=False, ncol=2, fontsize=8.5)
        fig.suptitle(f"{reader}: A's anchors land anywhere in the reader's ranking",
                     x=0.02, y=1.02, ha="left", fontsize=11, fontweight="bold")
        figs._finish(fig, f"fig12b_percentile_curves_{reader}", rows,
                     ["reader", "split", "problem_id", "chunk_idx", "abs_importance_B", "percentile",
                      "role", "n_candidates"])


if __name__ == "__main__":
    fig_anchor_strip()
    fig_percentile_curves()
