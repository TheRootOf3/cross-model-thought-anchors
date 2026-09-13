#!/usr/bin/env python
"""Regenerate every figure from runs/. One function per figure; each
writes plots/<name>.png and <name>_data.csv (exactly what is plotted). The
caption lives in the docstring. Every major experiment ends with one figure.

    python scripts/90_make_figures.py [--only fig0]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.answers import p_correct
from src.common.data import ROLLOUTS_ROOT, Trace
from src.common.prompts import block_flags
from src.common.io import REPO_ROOT, read_rows, rng

PLOTS = REPO_ROOT / "plots"
SURFACE, GRID, INK, MUTED = "#fcfcfb", "#e8e8e4", "#2b2b2b", "#6b6b66"


def _style():
    """Inter (SIL OFL, plots/fonts/) if present, else DejaVu; light surface."""
    from matplotlib import font_manager as fm
    for f in (PLOTS / "fonts").glob("Inter-*.ttf"):
        fm.fontManager.addfont(str(f))
    fam = "Inter" if any(x.name == "Inter" for x in fm.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": fam, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#c9c9c4", "axes.linewidth": 0.8})
# dataviz palette, validated 2026-09-08 (light surface): blue, orange, aqua.
COND = {  # run-id suffix -> (label, hex)
    "":        ("A's sampling (T 0.6, top-p 0.95), 16k cap",        "#2a78d6"),
    "_rec":    ("recommended sampling, 16k",   "#eb6834"),
    "_rec32k": ("recommended sampling, 32k",   "#1baf7a"),
}
MODELS = [("r1-distill-14b", "A: R1-Distill\n14B"), ("qwen3.5-9b", "Qwen3.5\n9B"),
          ("qwen3.5-4b", "Qwen3.5\n4B"), ("gpt-oss-20b", "GPT-OSS\n20B"), ("olmo3-7b-think", "Olmo-3\n7B-Think")]
PROBLEMS = [("", "correct_base_solution/problem_1591", "Problem 1591 · compound interest", "A's own trace was correct"),
            ("_330", "incorrect_base_solution/problem_330", "Problem 330 · nested 3(1+3(1+…))", "A's own trace was wrong")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


RUN_SUFFIX, FIG_TAG, N_LABEL = "", "", "8"   # set by --suffix: e.g. "_n32" -> runs probe_*_n32, figures fig1*


def probe_summary(run_id):
    """Answered-only P(correct), the convention thought-anchors uses for every
    accuracy it computes and the one configs/phase1.json declares - applied to
    readers and to A alike. `reopened` is re-derived from the raw completion,
    because rows written before 2026-09-08 carry a stale gpt-oss flag."""
    rows = list(read_rows(REPO_ROOT / "runs" / (run_id + RUN_SUFFIX) / "rows.jsonl"))
    if not rows:
        return None
    toks = sorted(r["tokens"] for r in rows)
    pc = p_correct(rows)
    reopened = sum(block_flags(r["model"], r["completion"], r["finish_reason"])["reopened"]
                   for r in rows)
    return {"n": len(rows), "n_answered": pc["n_answered"], "n_no_answer": pc["n_no_answer"],
            "correct": sum(bool(r["correct"]) for r in rows), "p_correct": pc["p_correct"],
            "capped": sum(bool(r["capped"]) for r in rows), "reopened": reopened,
            "tokens_median": toks[len(toks) // 2]}


def _probe_points():
    """One record per (model, problem, condition, prefilled) probe run on disk."""
    data = []
    for psuf, tname, *_ in PROBLEMS:
        for key, _ in MODELS:
            for csuf in COND:
                for pre, rsuf in ((True, csuf), (False, "_nopre32k" if csuf == "_rec32k" else None)):
                    if rsuf is None:
                        continue
                    s = probe_summary(f"probe_{key}{psuf}{rsuf}")
                    if s:
                        data.append({"run_id": f"probe_{key}{psuf}{rsuf}{RUN_SUFFIX}", "model": key, "problem": tname,
                                     "condition": COND[csuf][0], "cond_key": csuf, "prefilled": pre, **s})
    return data


def _a_reference(chunk=8):
    """A's released rollouts from the SAME starting point as the points drawn:
    chunk 8 for prefilled runs, chunk 0 for own-CoT runs. Answered-only, as
    everywhere else. Drawing the chunk-8 line against own-CoT points overstated
    A by 13 points on problem_1591 (0.870 vs 0.740)."""
    ref = {}
    for _, tname, *_ in PROBLEMS:
        t = Trace(ROLLOUTS_ROOT / tname, tname.split("/")[0])
        ans = [r for r in t.rollouts(chunk) if r.get("answer")]
        ref[tname] = wilson(sum(bool(r["is_correct"]) for r in ans), len(ans))  # (p, lo, hi)
    return ref


def _probe_grid(data, series, name, suptitle, subtitle, ref_chunks=(8,)):
    """2 rows (P(correct), fraction capped) x 2 columns (problems). `series` =
    list of (label, filter(d)->bool, hex, marker, filled, dodge)."""
    if not data:
        print(f"{name}: no probe runs on disk - skipped. scripts/01_prefill_probe.py writes them; "
              f"the grid the setup figure used is --suffix _n32.")
        return
    from matplotlib.ticker import PercentFormatter
    _style()
    refs = [(c, _a_reference(c)) for c in ref_chunks]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.26, "wspace": 0.16})
    x0 = {k: i for i, (k, _) in enumerate(MODELS)}
    for col, (psuf, tname, ptitle, psub) in enumerate(PROBLEMS):
        for row, metric in enumerate(("correct", "capped")):
            ax = axes[row][col]
            if row == 0:
                # One band per starting point actually plotted: A's released
                # rollouts from 8 prefilled sentences, and/or from chunk 0 for
                # the own-CoT points. Comparing own-CoT against the chunk-8
                # line overstated A by 13 points on problem_1591.
                subtitle_bits = []
                for chunk, ref in refs:
                    p, lo, hi = ref[tname]
                    what = "8 prefilled sentences" if chunk else "the problem alone"
                    ax.axhspan(lo, hi, color="#dedcd6", alpha=0.55 if chunk else 0.3, lw=0, zorder=0,
                               label=f"A's released rollouts from {what} (n = 100, 95% CI)" if col == 0 else None)
                    ax.axhline(p, color=MUTED, lw=0.9, ls=(0, (5, 4)) if chunk else (0, (1, 3)), zorder=1)
                    subtitle_bits.append(f"{p:.0%} " + ("prefilled" if chunk else "from the problem alone"))
                ax.set_title(f"{ptitle}\n\n", fontsize=10, fontweight="semibold", color=INK, loc="left", pad=6)
                ax.text(0, 1.03, f"{psub}\nA's released rollouts: " + "  ·  ".join(subtitle_bits),
                        transform=ax.transAxes, fontsize=8.3, color=MUTED, va="bottom", linespacing=1.35)
            for label, keep, hexc, marker, filled, dodge in series:
                pts = [d for d in data if d["problem"] == tname and keep(d)]
                if not pts:
                    continue
                xs = [x0[d["model"]] + dodge for d in pts]
                if metric == "capped":
                    # observed count of an exact quantity: plain bars, labelled with the count
                    frac = [d["capped"] / d["n"] for d in pts]
                    w = 0.22
                    ax.bar(xs, frac, w, color=hexc, lw=0, zorder=3)
                    if not filled:  # own-CoT bars keep the shape channel: ink outline
                        ax.bar(xs, frac, w, fill=False, edgecolor=INK, lw=0.9, zorder=4)
                    for x, f, d in zip(xs, frac, pts):
                        if d["capped"]:
                            ax.text(x, f + 0.03, str(d["capped"]), ha="center", va="bottom", fontsize=7, color=MUTED)
                    continue
                pts = [d for d in pts if d["n_answered"] > 0]
                if not pts:
                    continue
                ps = [wilson(d["correct"], d["n_answered"]) for d in pts]
                xs = [x0[d["model"]] + dodge for d in pts]
                # max(0, .): at p = 1 the Wilson bound is 1 - eps and hi - p is -0.0, which errorbar rejects
                ax.errorbar(xs, [q[0] for q in ps], yerr=[[max(0.0, q[0] - q[1]) for q in ps], [max(0.0, q[2] - q[0]) for q in ps]],
                            fmt="none", ecolor=hexc, elinewidth=1.7, capsize=0, alpha=0.85, zorder=2)
                ax.scatter(xs, [q[0] for q in ps], s=64 if filled else 58, marker=marker, zorder=4,
                           facecolor=hexc if filled else SURFACE, edgecolor=SURFACE if filled else INK, linewidth=1.4 if filled else 1.3,
                           label=label if (row, col) == (0, 0) else None)
            ax.set_ylim(-0.04, 1.06); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.grid(axis="y", color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=3, color="#c9c9c4")
        axes[1][col].set_xticks(list(x0.values())); axes[1][col].set_xticklabels([n for _, n in MODELS], linespacing=1.15)
        axes[1][col].set_xlim(-0.6, len(MODELS) - 0.4)
    axes[0][0].set_ylabel("P(correct | answered)", color=INK)
    axes[1][0].set_ylabel(f"stopped by the token cap (of {N_LABEL})", color=INK)
    from matplotlib.patches import Patch
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.04, 0.875), handletextpad=0.5,
               columnspacing=1.6, labelspacing=0.45)
    fig.text(0.045, 0.985, suptitle, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.945, subtitle, fontsize=9.2, color=MUTED, va="top", linespacing=1.45)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.68, bottom=0.10)
    fig.savefig(PLOTS / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.12); plt.close(fig)
    with open(PLOTS / f"{name}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(f"wrote plots/{name}.png + _data.csv  ({len(data)} points)")


def fig0a_sampling_cap():
    """Prefilled only: the pre-registered design sampling at 16k vs each model's recommended sampling at 16k and 32k."""
    data = [d for d in _probe_points() if d["prefilled"]]
    series = [(COND[c][0], (lambda d, c=c: d["cond_key"] == c), COND[c][1], "o", True, dodge)
              for c, dodge in (("", -0.27), ("_rec", 0.0), ("_rec32k", 0.27))]
    _probe_grid(data, series, f"fig{FIG_TAG or '0'}a_sampling_cap",
                "Sampling settings and token cap",
                f"Readers continue A's first 8 sentences. With each model's recommended sampling and a 32k cap, every continuation finishes.\nn = {N_LABEL} per point, Wilson 95% CI on P(correct | answered)",)


def fig0b_prefill_vs_own():
    """At the final settings (recommended sampling, 32k): continuing A vs the reader's own chain of thought."""
    data = [d for d in _probe_points() if d["cond_key"] == "_rec32k"]
    series = [("continuing A's first 8 sentences", lambda d: d["prefilled"], COND["_rec32k"][1], "o", True, -0.15),
              ("own chain of thought (problem only)", lambda d: not d["prefilled"], COND["_rec32k"][1], "D", False, 0.15)]
    _probe_grid(data, series, f"fig{FIG_TAG or '0'}b_prefill_vs_own",
                "Continuing A's reasoning vs the reader's own chain of thought",
                f"Recommended sampling, 32k cap. The Qwens finish only with the prefix; for gpt-oss and Olmo the intervals admit a change of ±0.3 either way.\nn = {N_LABEL} per point, Wilson 95% CI on P(correct | answered)",
                ref_chunks=(8, 0))


# ---------------------------------------------------------------- the anchor table: anchor table
ANCHOR, PARTNER, ALL = "#2a78d6", "#eb6834", "#b9b9b3"   # validated palette + neutral


def _anchor_table():
    import pandas as pd
    d = pd.read_csv(REPO_ROOT / "data/anchor_table.csv")
    t = d[d.testable == True]
    return d, t, d[d.anchor_rank.notna()], d[d.partner_of.notna()]


def _finish(fig, name, rows, header):
    fig.savefig(PLOTS / f"{name}.png", dpi=220, bbox_inches="tight")
    with open(PLOTS / f"{name}_data.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    plt.close(fig)
    print(f"wrote plots/{name}.png + _data.csv")


def fig2c_anchor_pairs_by_trace():
    """Figure 2c. The 120 anchor/partner pairs, one row per trace (20 problems ×
    correct/wrong A trace). Blue: the anchor's released counterfactual importance
    (accuracy); orange: its
    partner's; the grey link is the gap A's own scores show, the reference for
    what a reader's paired d would be if importance were entirely in the text.
    Anchors left of zero were needed by A; right of zero, A did better without.
    Counterfactual importance = P(correct | the sentence is resampled into something
    semantically different) − P(correct | the sentence is kept), in units of final-answer
    accuracy, from A's 100 released rollouts per sentence."""
    _style()
    d, t, a, p = _anchor_table()
    order = [(s, pid) for s in ("correct_base_solution", "incorrect_base_solution")
             for pid in sorted(d[d.split == s].problem_id.unique(), key=lambda x: int(x.split("_")[1]))]
    fig, ax = plt.subplots(figsize=(6.2, 8.2))
    rows = []
    pidx = {(r.split, r.problem_id, r.chunk_idx): r for r in p.itertuples()}
    for y, (s, pid) in enumerate(order):
        for r in a[(a.split == s) & (a.problem_id == pid)].itertuples():
            yy = y + (r.anchor_rank - 2) * 0.26
            q = next(v for k, v in pidx.items() if k[0] == s and k[1] == pid and v.partner_of == r.chunk_idx)
            ax.plot([q.counterfactual_importance_accuracy, r.counterfactual_importance_accuracy], [yy, yy], color="#cfcfca", lw=1, zorder=1)
            ax.scatter(r.counterfactual_importance_accuracy, yy, s=16, color=ANCHOR, zorder=3, linewidths=0)
            ax.scatter(q.counterfactual_importance_accuracy, yy, s=16, color=PARTNER, zorder=3, linewidths=0)
            rows.append((s, pid, int(r.anchor_rank), int(r.chunk_idx), round(r.counterfactual_importance_accuracy, 4),
                         int(q.chunk_idx), round(q.counterfactual_importance_accuracy, 4)))
    ax.axvline(0, color=INK, lw=0.6, alpha=0.5)
    ax.axhline(19.5, color=GRID, lw=1)
    ax.text(-0.72, 9.5, "A's trace\nwas correct", ha="left", va="center", color=MUTED, fontsize=8.2)
    ax.text(-0.72, 29.5, "A's trace\nwas wrong", ha="left", va="center", color=MUTED, fontsize=8.2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([pid.replace("problem_", "") for _, pid in order], fontsize=7.5)
    ax.set_ylim(len(order) - 0.5, -0.5); ax.set_xlim(-0.75, 1.3)
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("A's counterfactual importance (accuracy)")
    ax.set_ylabel("problem")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.set_title("The 120 anchor/partner pairs, by trace", loc="left", fontweight="bold")
    # below the axes: the data fills the upper right, where an in-plot legend sat on it
    ax.scatter([], [], color=ANCHOR, s=22, label="anchor  (top 3 per trace by |importance|, in the trace's direction)")
    ax.scatter([], [], color=PARTNER, s=22, label="partner  (within ±4 of its anchor, importance nearest zero)")
    ax.plot([], [], color="#cfcfca", lw=1.4, label="the gap A's own scores show")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.055), fontsize=8.2,
              handletextpad=0.6, labelspacing=0.55, ncol=1)
    _finish(fig, "fig2c_anchor_pairs_by_trace", rows,
            ["split", "problem_id", "anchor_rank", "anchor_chunk", "anchor_counterfactual_importance_accuracy", "partner_chunk", "partner_counterfactual_importance_accuracy"])


# ---------------------------------------------------------------- the reader screen: saturation screen
READERS = [("qwen3.5-9b", "Qwen3.5-9B (B_near candidate)"), ("qwen3.5-4b", "Qwen3.5-4B (B_near fallback)"),
           ("qwen3-4b-thinking-2507", "Qwen3-4B-Thinking-2507 (B_near candidate, added)"), ("qwen3.5-2b", "Qwen3.5-2B (B_near candidate, added)"),
           ("qwen3-1.7b", "Qwen3-1.7B (B_near candidate, tier 2)"), ("qwen3.5-0.8b", "Qwen3.5-0.8B (B_near candidate, tier 2)"),
           ("gpt-oss-20b", "GPT-OSS-20B (B_far)"), ("olmo3-7b-think", "Olmo-3-7B-Think (B_far fallback)")]


def _screen_points():
    """Per reader x problem from runs/screen_<reader>/rows.jsonl: answered-only
    P(correct) with Wilson CI, cap hits, re-opens; plus A's released accuracy
    from the problem alone (chunk-0 rollouts of BOTH splits pooled, ~200)."""
    import json
    cfg = json.load(open(REPO_ROOT / "configs/phase1.json"))["saturation_screen"]
    problems = sorted((ROLLOUTS_ROOT / "correct_base_solution").glob("problem_*"), key=lambda p: int(p.name.split("_")[1]))
    a_ref = {}
    for d in problems:
        # A from the problem alone = the chunk-0 rollouts (empty prefix) of BOTH
        # splits pooled: 2 x 100 independent draws under the dataset's sampling
        # (T 0.6 / top-p 0.95 / 16k); the split only selects the base solution,
        # not these resamples. Answered-only, as everywhere.
        ans = [r for split in ("correct_base_solution", "incorrect_base_solution")
               for r in Trace(ROLLOUTS_ROOT / split / d.name, split).rollouts(0) if r.get("answer")]
        a_ref[d.name] = wilson(sum(str(r["is_correct"]).lower() == "true" for r in ans), len(ans)) + (len(ans),)
    out = []
    for key, label in READERS:
        if not (REPO_ROOT / f"runs/screen_{key}/rows.jsonl").exists():
            continue
        rows = list(read_rows(REPO_ROOT / f"runs/screen_{key}/rows.jsonl"))
        for d in problems:
            rs = [r for r in rows if r["problem_id"] == d.name]
            pc = p_correct(rs) if rs else {"p_correct": None, "n_answered": 0, "n_no_answer": 0}
            k = sum(bool(r["correct"]) for r in rs if r.get("answer"))
            p, lo, hi = wilson(k, pc["n_answered"]) if pc["n_answered"] else (float("nan"),) * 3
            out.append({"reader": key, "reader_label": label, "problem_id": d.name, "n": len(rs),
                        "n_answered": pc["n_answered"], "n_correct": k, "p_correct": p, "ci_lo": lo, "ci_hi": hi,
                        "n_capped": sum(bool(r["capped"]) for r in rs), "n_reopened": sum(bool(r["reopened"]) for r in rs),
                        "in_band": bool(pc["n_answered"]) and cfg["band_low"] <= p <= cfg["band_high"],
                        "a_p_correct": a_ref[d.name][0], "a_ci_lo": a_ref[d.name][1], "a_ci_hi": a_ref[d.name][2], "a_n_answered": a_ref[d.name][3]})
    return out, cfg


def fig3e_screen_selected():
    """Figure 3e. The saturation screen for the two readers Phase 1 actually used -
    the same measurement as figure 3, with the six candidates that failed the band
    left out. Each reader answers each of the 20 problems from the problem alone,
    40 samples, its own recommended sampling, 32k cap. Points: P(correct | answered)
    with Wilson 95% CI; filled = inside the 15-85% band, open = saturated. Grey: A's
    released rollouts from the same problems with no prefix (chunk-0 of both splits
    pooled, n ~ 200). Problems ordered by A's accuracy; cap hits under the axis."""
    fig3_screen(only=("gpt-oss-20b", "qwen3-1.7b"), name="fig3e_screen_selected",
                title="Phase 1 reader selection: accuracy on MATH tasks",
                subtitle="Readers with >= 10 of 20 problems inside the 15-85% band. "
                         "40 samples per problem; Wilson 95% CI; 32k token-cap hits under the axis")


def fig3_screen(only=None, name="fig3_screen", title="Saturation screen: readers on the 20 problems, no prefix",
                subtitle=None):
    """Figure 3. Saturation screen: each reader answers each of the 20 problems
    from the problem alone, 40 samples, its own recommended sampling, 32k cap.
    Points: P(correct | answered) with Wilson 95% CI; filled = inside the
    15-85% band (kept for Phase 1), open = saturated. Grey: A's released
    rollouts from the problem alone (chunk-0 rollouts of both splits pooled,
    ~200, Wilson CI). Problems ordered by A's
    accuracy. Small numbers under the axis: rollouts stopped by the cap."""
    import numpy as np
    _style()
    data, cfg = _screen_points()
    order = sorted({d["problem_id"] for d in data}, key=lambda p: next(x["a_p_correct"] for x in data if x["problem_id"] == p))
    x0 = {p: i for i, p in enumerate(order)}
    selected = {}   # reader -> {problem_id: "correct" | "incorrect"}; trace sets are per reader
    tf = REPO_ROOT / "data/phase1_traces.json"
    if only and tf.exists():
        for rk, blk in json.load(open(tf))["by_reader"].items():
            selected[rk] = {t["problem_id"]: ("correct" if t["split"].startswith("correct") else "incorrect")
                            for t in blk["traces"]}
    readers = [(k, l) for k, l in READERS if any(d["reader"] == k for d in data) and (only is None or k in only)]
    if only:   # keep the requested order, and drop the candidate-status suffix from the label
        readers = [(k, dict(READERS)[k].split(" (")[0]) for k in only]
        data = [d for d in data if d["reader"] in only]
    nrow = (len(readers) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 3.3 * nrow), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.1})
    for ax, (key, label) in zip(axes.flat, readers):
        pts = [d for d in data if d["reader"] == key]
        n_rows = sum(d["n"] for d in pts)
        ax.axhspan(cfg["band_low"], cfg["band_high"], color="#e9e7e1", lw=0, zorder=0)
        for d in pts:
            x = x0[d["problem_id"]]
            ax.errorbar(x - 0.18, d["a_p_correct"], yerr=[[d["a_p_correct"] - d["a_ci_lo"]], [max(0, d["a_ci_hi"] - d["a_p_correct"])]],
                        fmt="o", ms=4.5, color="#b9b9b3", ecolor="#b9b9b3", elinewidth=1.2, capsize=0, zorder=2)
            if d["n_answered"]:
                ax.errorbar(x + 0.18, d["p_correct"], yerr=[[max(0, d["p_correct"] - d["ci_lo"])], [max(0, d["ci_hi"] - d["p_correct"])]],
                            fmt="none", ecolor=ANCHOR, elinewidth=1.6, capsize=0, zorder=3)
                ax.scatter(x + 0.18, d["p_correct"], s=46, zorder=4, facecolor=ANCHOR if d["in_band"] else SURFACE,
                           edgecolor=ANCHOR, linewidth=1.4)
            if d["n_capped"]:
                ax.text(x + 0.18, -0.09, str(d["n_capped"]), ha="center", va="top", fontsize=6.5, color=MUTED, clip_on=False)
        n_in = sum(d["in_band"] for d in pts)
        ax.set_title(f"{label}   ·   {n_in}/{len(order)} in band   ·   {n_rows} rollouts", loc="left", fontweight="bold", pad=8)
        ax.set_ylim(-0.04, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_xticks(range(len(order))); ax.set_xticklabels([p.replace("problem_", "") for p in order], rotation=90, fontsize=7.5)
        used = selected.get(key, {})
        for lab, pid in zip(ax.get_xticklabels(), order):
            if pid in used:            # same convention as the paired figures: blue = A's trace correct
                lab.set_color(ANCHOR if used[pid] == "correct" else PARTNER)
                lab.set_fontweight("bold")
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in axes[:, 0]: ax.set_ylabel("P(correct | answered)")
    for ax in axes[-1]:
        ax.set_xlabel("problem (ordered by A's accuracy from the problem alone)", labelpad=10)
        ax.tick_params(axis="x", pad=18)   # cap-hit counts sit between axis and labels
    axes[0][0].scatter([], [], s=46, facecolor=ANCHOR, edgecolor=ANCHOR, label="reader, in band (15-85%)")
    axes[0][0].scatter([], [], s=46, facecolor=SURFACE, edgecolor=ANCHOR, linewidth=1.4, label="reader, saturated")
    axes[0][0].scatter([], [], s=30, color="#b9b9b3", label="A's released rollouts from the problem alone (both splits pooled, n ≈ 200)")
    if only:
        for c, lab in ((ANCHOR, "problem label: used as a trace A got CORRECT"), (PARTNER, "used as a trace A got WRONG")):
            axes[0][0].plot([], [], marker="$\\mathbf{123}$", color=c, ls="none", ms=13, label=lab)
    H = 3.3 * nrow   # figure height in inches; offsets below are in inches
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.06, 1.0 + 0.34 / H), ncol=3, frameon=False)
    fig.text(0.06, 1.0 + 0.98 / H, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.06, 1.0 + 0.72 / H, subtitle or ("40 samples per problem, each reader's recommended sampling, 32k cap; "
             "Wilson 95% CI on the answered rollouts; cap hits printed under the axis"),
             fontsize=9.2, color=MUTED, va="top")
    fig.subplots_adjust(top=1.0 - (0.62 if only else 0.35) / H, bottom=0.9 / H, left=0.06, right=0.99)
    fig.savefig(PLOTS / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
    with open(PLOTS / f"{name}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(f"wrote plots/{name}.png + _data.csv  ({len(data)} reader x problem points)")


DETAIL_TRACES = [("correct_base_solution", "problem_4019"), ("incorrect_base_solution", "problem_330")]


def fig3d_two_traces():
    """Figure 3d. Two of the traces selected for GPT-OSS-20B in detail, one per
    column: problem 4019 (A's trace was correct) and problem 330 (A's trace was
    wrong). Top: A's
    released accuracy when restarted at each sentence, P(correct | answered)
    over ~100 rollouts, with its Wilson 95% band; the shaded tail after the
    convergence index is not tested. Bottom: A's released counterfactual
    importance (accuracy) per sentence, P(correct | replaced) − P(correct | kept),
    both directions drawn; the rule considers only the negative ones on the correct
    trace and the positive ones on the wrong one (full tone), the other direction is
    light grey; reddish where the sentence is excluded (overdetermined,
    unmeasured, corrupted prefix, or after the convergence index); blue = the three
    anchors (top-3 by |importance| among the eligible ones; rank shown), orange diamonds = their partners, joined to their anchor. Vertical
    guides carry the anchor positions into the accuracy panel."""
    import json
    import pandas as pd
    _style()
    sel = {(t["split"], t["problem_id"]): t for t in json.load(open(REPO_ROOT / "data/phase1_traces.json"))["traces"]}
    tab = pd.read_csv(REPO_ROOT / "data/anchor_table.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.4), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.12, "wspace": 0.14})
    rows = []
    for col, (split, pid) in enumerate(DETAIL_TRACES):
        t = sel[(split, pid)]
        g = tab[(tab.split == split) & (tab.problem_id == pid)].sort_values("chunk_idx")
        n, conv = t["n_chunks"], g.convergence_idx.iloc[0]
        conv = None if pd.isna(conv) else int(conv)
        idx, imp, acc, test = g.chunk_idx.values, g.counterfactual_importance_accuracy.values, g.accuracy.values, g.testable.values.astype(bool)
        tr = Trace(ROLLOUTS_ROOT / split / pid, split)
        n_ans = [sum(1 for r in tr.rollouts(i) if r.get("answer")) for i in idx]
        lo = [wilson(round(a * m), m)[1] if m else float("nan") for a, m in zip(acc, n_ans)]
        hi = [wilson(round(a * m), m)[2] if m else float("nan") for a, m in zip(acc, n_ans)]
        top, bot = axes[0][col], axes[1][col]
        for ax in (top, bot):
            if conv is not None:
                ax.axvspan(conv - 0.5, n - 0.5, color="#f1f1ed", lw=0, zorder=0)
                ax.axvline(conv - 0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
            for pr in t["pairs"]:
                ax.axvline(pr["anchor_chunk"], color=ANCHOR, lw=0.7, alpha=0.35, zorder=1)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        # top: accuracy line + Wilson band
        top.fill_between(idx, lo, hi, color="#cfcfca", alpha=0.5, lw=0, zorder=2)
        top.plot(idx, acc, color=INK, lw=1.3, zorder=3)
        for pr in t["pairs"]:
            a = pr["anchor_chunk"]
            top.scatter(a, acc[a], s=34, color=ANCHOR, zorder=5, linewidths=0)
            top.scatter(pr["partner_chunk"], acc[pr["partner_chunk"]], s=30, marker="D", color=PARTNER, zorder=5, linewidths=0)
        top.set_ylim(-0.03, 1.05); top.set_yticks([0, 0.25, 0.5, 0.75, 1]); top.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        if conv is not None:
            top.text(n - 1.5, 0.97, f"locked from sentence {conv}", fontsize=7.5, color=MUTED, va="top", ha="right")
        top.set_title(f"Problem {pid.split('_')[1]} · A's trace was {'correct' if split.startswith('correct') else 'wrong'} · {n} sentences",
                      loc="left", fontweight="bold", pad=8)
        # bottom: importance bars, anchors, partners
        want = -1 if split.startswith("correct") else 1
        same = imp * want > 0                                            # the rule considers this direction only; the other is drawn light
        bot.vlines(idx[test & same], 0, imp[test & same], color="#b3b3ad", lw=1.2, zorder=2)
        bot.vlines(idx[~test & same], 0, imp[~test & same], color="#d6aaa2", lw=1.2, zorder=2)
        bot.vlines(idx[~same], 0, imp[~same], color="#e2e2dd", lw=1.2, zorder=2)
        bot.axhline(0, color=INK, lw=0.6, alpha=0.6, zorder=3)
        roles = {}
        for pr in t["pairs"]:
            a, q, ai, qi = pr["anchor_chunk"], pr["partner_chunk"], pr["anchor_cf_importance"], pr["partner_cf_importance"]
            bot.plot([q, a], [qi, ai], color=PARTNER, lw=0.9, alpha=0.8, zorder=4)
            bot.vlines(a, 0, ai, color=ANCHOR, lw=3, zorder=5)
            bot.scatter(a, ai, s=40, color=ANCHOR, zorder=6, linewidths=0)
            bot.scatter(q, qi, s=34, marker="D", color=PARTNER, zorder=6, linewidths=0)
            bot.text(a + 1.2, ai, f"#{pr['anchor_rank']}", ha="left", va="center", fontsize=8, color=ANCHOR, fontweight="bold")
            roles[a] = f"anchor {pr['anchor_rank']}"; roles[q] = f"partner of {a}"
        bot.set_ylim(-0.75, 0.95); bot.set_yticks([-0.5, 0, 0.5]); bot.set_yticklabels(["−0.5", "0", "+0.5"])
        bot.set_xlim(-1.5, n + 0.5); bot.set_xlabel("sentence index in A's trace")
        for i, v, ac, lo_, hi_, m, ts in zip(idx, imp, acc, lo, hi, n_ans, test):
            rows.append((split, pid, int(i), round(float(v), 4), round(float(ac), 4), round(lo_, 4), round(hi_, 4), m, bool(ts), conv, roles.get(int(i), "")))
    axes[0][0].set_ylabel("A's accuracy restarted\nat this sentence")
    axes[1][0].set_ylabel("A's counterfactual\nimportance (accuracy)")
    h = [plt.Line2D([], [], color=INK, lw=1.3, label="A's released accuracy per sentence, P(correct | answered), ~100 rollouts, Wilson 95% band"),
         plt.Line2D([], [], color=ANCHOR, lw=3, marker="o", ms=6, label="anchor (top-3 by |counterfactual importance|, direction matching the outcome)"),
         plt.Line2D([], [], color=PARTNER, lw=0.9, marker="D", ms=5, label="partner (closest to zero within ±4)"),
         plt.Line2D([], [], color="#b3b3ad", lw=3, label="counterfactual importance, eligible sentences (the outcome's direction only)"),
         plt.Line2D([], [], color="#d6aaa2", lw=3, label="excluded (e.g. overdetermined, corrupted prefix, locked)"),
         plt.Line2D([], [], color="#e2e2dd", lw=3, label="opposite direction (shown, not considered by the rule)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.06, 0.995), ncol=2, frameon=False, fontsize=7.8)
    fig.text(0.06, 1.05, "Case study: two of the traces selected for GPT-OSS-20B", fontsize=13, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.09)
    _finish(fig, "fig3d_two_traces", rows,
            ["split", "problem_id", "chunk_idx", "counterfactual_importance_accuracy", "accuracy", "acc_ci_lo", "acc_ci_hi", "n_answered", "testable", "convergence_idx", "role"])


def fig2d_pair_examples(reader="gpt-oss-20b", per_group=3, seed=20260908):
    """Figure 2d. Six anchor/partner pairs drawn at random, three from traces A got right
    and three from traces it got wrong, (seeded, src.common.io.rng)
    from the reader's pairs with a defined d, to show what a single d is made of. The
    sentences are quoted verbatim from A's trace, left; on the right, counterfactual
    importance - P(correct | the sentence is resampled to something dissimilar) −
    P(correct | it is kept), in accuracy - for A's released measurement and for the
    reader, anchor in blue and partner in orange. A's anchor is far from zero and its
    partner near it by construction: the pair was chosen on A's released data alone,
    before any reader rollout existed. d is the reader's gap between the two, signed by
    A's direction at the anchor, and is the paired statistic the gate reads. Sentences
    and numbers are read from data/p1_importance.csv and data/p1_pairs.csv."""
    import csv as _csv
    import textwrap
    _style()
    sent = {(r["reader"], r["problem_id"], int(r["chunk_idx"])): r
            for r in _csv.DictReader(open(REPO_ROOT / "data/p1_importance.csv"))}
    pairs = [r for r in _csv.DictReader(open(REPO_ROOT / "data/p1_pairs.csv"))
             if r["reader"] == reader and r["d"]]
    groups = [("A's trace was correct", "correct"), ("A's trace was wrong", "incorrect")]
    picked, bands = [], []
    for label, split in groups:
        pool, r_ = [r for r in pairs if r["split"].startswith(split)], rng(seed, "fig2d_pair_examples", reader, split)
        by_problem = {}
        for r in pool:
            by_problem.setdefault(r["problem_id"], []).append(r)
        chosen = sorted((r_.choice(by_problem[pid]) for pid in r_.sample(sorted(by_problem), per_group)),
                        key=lambda r: (r["problem_id"], int(r["anchor_chunk"])))
        bands.append((label, len(picked), len(picked) + len(chosen) - 1))
        picked += chosen
    k = len(picked)
    fig = plt.figure(figsize=(13.4, 2.65 * k + 1.2))
    gs = fig.add_gridspec(k, 2, width_ratios=[1.45, 1], hspace=0.62, wspace=0.02,
                          left=0.105, right=0.985, top=0.92, bottom=0.085)
    lim = 1.15 * max(abs(float(sent[(reader, pr["problem_id"], int(pr[f"{role}_chunk"]))][f]))
                     for pr in picked for role in ("anchor", "partner")
                     for f in ("importance_A", "importance_B"))
    rows, text_axes = [], []
    for row, pr in enumerate(picked):
        pid, a_chunk, p_chunk = pr["problem_id"], int(pr["anchor_chunk"]), int(pr["partner_chunk"])
        a, b = sent[(reader, pid, a_chunk)], sent[(reader, pid, p_chunk)]
        correct = pr["split"].startswith("correct")
        d, sign = float(pr["d"]), int(pr["direction"])

        txt = fig.add_subplot(gs[row, 0]); txt.axis("off")
        txt.text(0, 1.06, f"Problem {pid.split('_')[1]}", fontsize=11.5, fontweight="bold",
                 va="top", transform=txt.transAxes)
        text_axes.append(txt)
        y = 0.86                                     # stacked, each block placed under the rendered height of the last
        for s_, col, label in ((a, ANCHOR, f"anchor · sentence {a_chunk}"),
                               (b, PARTNER, f"partner · sentence {p_chunk}")):
            body = textwrap.fill(s_["text"].replace("\n", " "), 54)
            txt.text(0.012, y, label, fontsize=10, color=col, fontweight="bold", va="bottom",
                     transform=txt.transAxes)
            t = txt.text(0.012, y - 0.05, body, fontsize=12, color=INK, va="top", transform=txt.transAxes,
                         linespacing=1.4,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=col, linewidth=0.9, alpha=0.9))
            fig.canvas.draw()                        # the box height is only known once it is rendered
            bb = t.get_bbox_patch().get_window_extent(renderer=fig.canvas.get_renderer())
            y = txt.transAxes.inverted().transform((0, bb.y0))[1] - 0.12

        ax = fig.add_subplot(gs[row, 1])
        for y0, field in ((1, "importance_A"), (0, "importance_B")):
            for s_, col, off in ((a, ANCHOR, 0.17), (b, PARTNER, -0.17)):
                v = float(s_[field])
                ax.hlines(y0 + off, 0, v, color=col, lw=2.0, alpha=0.85, zorder=3)
                ax.plot(v, y0 + off, "o", color=col, ms=7, zorder=4,
                        mfc="none" if s_["short"] == "True" and field == "importance_B" else col)
                ax.annotate(f"{v:+.3f}", (v, y0 + off), textcoords="offset points",
                            xytext=(9 if v >= 0 else -9, 0), ha="left" if v >= 0 else "right",
                            va="center", fontsize=9.5, color=col)
        ax.axvline(0, color=INK, lw=0.8, alpha=0.7, zorder=2)
        ax.set_yticks([1, 0]); ax.set_yticklabels(["A (released)", f"{reader} (reader)"], fontsize=10.5)
        ax.set_ylim(-0.55, 1.55); ax.set_xlim(-lim, lim)
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if row == k - 1:
            ax.set_xlabel("counterfactual importance  (accuracy)", fontsize=11)
        ax.set_title(f"d = {d:+.3f}", loc="right", fontsize=11.5, color=INK, fontweight="bold", pad=6)
        for s_, role in ((a, "anchor"), (b, "partner")):
            rows.append((reader, pr["split"], pid, role, int(s_["chunk_idx"]), s_["tags"], s_["text"],
                         round(float(s_["importance_A"]), 4), round(float(s_["importance_B"]), 4),
                         int(s_["kept_n_answered"]), int(s_["replaced_n_answered"]), int(s_["replaced_n"]),
                         s_["short"] == "True", sign, round(d, 4)))
    fig.canvas.draw()
    for label, first, last in bands:                 # one bracket per outcome group, down the left margin
        top_y = text_axes[first].get_position().y1
        bot_y = text_axes[last].get_position().y0
        fig.lines.append(plt.Line2D([0.055, 0.055], [bot_y, top_y], color=MUTED, lw=1.0,
                                    transform=fig.transFigure, figure=fig))
        fig.text(0.045, (top_y + bot_y) / 2, label, rotation=90, ha="right", va="center",
                 fontsize=11.5, color=INK, fontweight="bold")
    fig.suptitle(f"{k} anchor/partner pairs sampled at random from A's chains of thought",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    h = [plt.Line2D([], [], color=ANCHOR, lw=2, marker="o", ms=6), plt.Line2D([], [], color=PARTNER, lw=2, marker="o", ms=6)]
    l = ["anchor: |importance| top-3 for A, in the direction of the trace's outcome",
         "partner: importance nearest zero for A, within 4 sentences of the anchor"]
    if any(r[-3] for r in rows):                  # only if a drawn pile actually lost rollouts
        h.append(plt.Line2D([], [], color=MUTED, lw=0, marker="o", ms=6, mfc="none"))
        l.append("hollow = a pile with missing rollouts")
    fig.legend(h, l,
               loc="upper right", bbox_to_anchor=(0.985, 0.052), frameon=False, ncol=1, fontsize=10,
               labelspacing=0.5, alignment="left")
    _finish(fig, "fig2d_pair_examples", rows,
            ["reader", "split", "problem_id", "role", "chunk_idx", "tags", "text", "importance_A", "importance_B",
             "kept_n_answered", "replaced_n_answered", "replaced_n_sampled", "short_pile", "sign_A", "d"])


def fig7b_dense_pair(pids=("problem_4605", "problem_2236")):
    """Figure 7b. Two dense profiles side by side, gpt-oss-20b restarted 100 times at
    every sentence of A's trace. Top row: accuracy, P(correct | answered), A's released
    curve (~100 rollouts, grey) against the reader's (blue), Wilson 95% bands; shading
    marks the sentences after A's convergence index, where A's answer is already fixed.
    Bottom row: counterfactual importance per sentence, P(correct | dissimilar at i) −
    accuracy(i+1), A's released (grey stems, left) against the reader's (blue stems,
    right); vertical guides are A's three anchors, open diamonds the reader's own top-3
    by |importance| in the direction matching the trace outcome.
    Left (problem 4605): the accuracy curves move together, pre-convergence means 0.79
    (A) against 0.83 (reader), and they share the dip at sentence 50. Right (2236): they
    come apart at sentence 24 where A commits to its error - over the rest of the
    pre-convergence trace A averages 0.06 (max 0.16) while the reader averages 0.70
    (min 0.38), so the prefix that locks A into a wrong answer still leaves the reader
    able to recover. In both traces the importance rows disagree about which sentences
    matter: A's anchors are not where the reader's diamonds fall. Whether accuracy
    tracks is a property of the trace; where importance sits is not."""
    import pandas as pd
    _style()
    tab = pd.read_csv(REPO_ROOT / "data/anchor_table.csv")
    df = pd.concat([pd.read_csv(f) for f in sorted((REPO_ROOT / "runs").glob("dense_*/profile.csv"))], ignore_index=True)
    reader = df.reader.iloc[0]
    sel = {(t["split"], t["problem_id"]): t for t in json.load(open(REPO_ROOT / "data/phase1_traces.json"))["by_reader"][reader]["traces"]}
    fig, axes = plt.subplots(2, len(pids), figsize=(5.9 * len(pids), 6.2), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.10, "wspace": 0.07})
    rows = []
    for col, pid in enumerate(pids):
        top, bot = axes[0][col], axes[1][col]
        g = df[df.problem_id == pid].sort_values("chunk_idx")
        split = g.split.iloc[0]
        t = sel[(split, pid)]
        conv = tab[(tab.split == split) & (tab.problem_id == pid)].convergence_idx.iloc[0]
        conv = None if pd.isna(conv) else int(conv)
        n, tr = len(g), Trace(ROLLOUTS_ROOT / split / pid, split)
        want = -1 if split.startswith("correct") else 1
        idx, a_acc, b_acc = g.chunk_idx.values, g.released_accuracy.values, g.accuracy.values
        a_imp, b_imp = g.released_cf_importance.values, g.cf_importance.values
        a_n = [sum(1 for r in tr.rollouts(i) if r.get("answer")) for i in idx]
        band = lambda acc, ns: ([wilson(round(a * m), m)[1] if m and not pd.isna(a) else float("nan") for a, m in zip(acc, ns)],
                                [wilson(round(a * m), m)[2] if m and not pd.isna(a) else float("nan") for a, m in zip(acc, ns)])
        a_lo, a_hi = band(a_acc, a_n)
        b_lo, b_hi = band(b_acc, g.n_answered.values)
        for ax in (top, bot):
            if conv is not None:
                ax.axvspan(conv - 0.5, n - 0.5, color="#f1f1ed", lw=0, zorder=0)
                ax.axvline(conv - 0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
            for pr in t["pairs"]:
                ax.axvline(pr["anchor_chunk"], color=ANCHOR, lw=0.7, alpha=0.35, zorder=1)
            ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        top.fill_between(idx, a_lo, a_hi, color="#cfcfca", alpha=0.5, lw=0, zorder=2)
        top.plot(idx, a_acc, color=MUTED, lw=1.2, zorder=3)
        top.fill_between(idx, b_lo, b_hi, color=ANCHOR, alpha=0.15, lw=0, zorder=2)
        top.plot(idx, b_acc, color=ANCHOR, lw=1.4, marker=".", ms=3, zorder=4)
        top.set_ylim(-0.03, 1.05); top.set_yticks([0, 0.25, 0.5, 0.75, 1]); top.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        top.set_title(f"Problem {pid.split('_')[1]} · A's trace was {'correct' if split.startswith('correct') else 'wrong'} · {n} sentences",
                      loc="left", fontsize=9.5)
        bot.vlines(idx - 0.2, 0, a_imp, color="#b3b3ad", lw=1.2, zorder=2)
        bot.vlines(idx + 0.2, 0, b_imp, color=ANCHOR, lw=1.2, zorder=3)
        bot.axhline(0, color=INK, lw=0.6, alpha=0.6, zorder=3)
        for pr in t["pairs"]:
            bot.text(pr["anchor_chunk"] + 1.0, 1.0 - 0.13 * (pr["anchor_rank"] - 1), f"A #{pr['anchor_rank']}",
                     ha="left", va="top", fontsize=7.5, color=ANCHOR, fontweight="bold")
        ok = ~pd.isna(a_imp) & ~pd.isna(b_imp) & (idx < (conv if conv is not None else n))
        cand = g[(g.cf_importance * want > 0) & ok].assign(mag=lambda d: d.cf_importance.abs())
        own = cand.sort_values(["mag", "chunk_idx"], ascending=[False, True]).head(3)      # ties: lower index first, as in 02
        bot.scatter(own.chunk_idx + 0.2, own.cf_importance, s=46, marker="D", facecolors="none",
                    edgecolors=ANCHOR, linewidths=1.3, zorder=6)
        bot.set_ylim(-1.05, 1.05); bot.set_yticks([-1, -0.5, 0, 0.5, 1]); bot.set_yticklabels(["−1", "−0.5", "0", "+0.5", "+1"])
        bot.set_xlim(-1.5, n + 0.5); bot.set_xlabel("sentence index in A's trace")
        top3 = set(own.chunk_idx)
        rows += [(reader, split, pid, int(i), *[None if pd.isna(v) else round(float(v), 4) for v in (aa, alo, ahi, ba, blo, bhi, ai, bi)],
                  int(na), int(nb), int(nd), r if isinstance(r, str) else "", int(i) in top3, conv)
                 for i, aa, alo, ahi, ba, blo, bhi, ai, bi, na, nb, nd, r in
                 zip(idx, a_acc, a_lo, a_hi, b_acc, b_lo, b_hi, a_imp, b_imp, a_n, g.n_answered.values,
                     g.n_dissimilar_answered.values, g.role.values)]
    axes[0][0].set_ylabel("accuracy when restarted\nat this sentence")
    axes[1][0].set_ylabel("counterfactual importance\nP(correct | dissimilar) − acc(i+1)")
    h = [plt.Line2D([], [], color=MUTED, lw=1.5), plt.Line2D([], [], color=ANCHOR, lw=1.5, marker=".", ms=5),
         plt.Rectangle((0, 0), 1, 1, color="#e6e6e1"), plt.Line2D([], [], color=ANCHOR, lw=0.9, alpha=0.5),
         plt.Line2D([], [], color=ANCHOR, marker="D", ls="", mfc="none")]
    fig.legend(h, ["A (released, ~100 rollouts)", f"{reader} (ours, 100 rollouts)", "after A's convergence index",
                   "A's anchors (#1-#3)", f"{reader}'s own top-3 by |importance|"],
               loc="upper left", bbox_to_anchor=(0.062, 0.045), frameon=False, ncol=3, fontsize=9, columnspacing=2.4)
    fig.suptitle("The reader's accuracy sometimes follows A's along the trace, and sometimes does not",
                 x=0.062, ha="left", fontsize=11.5, fontweight="bold")
    _finish(fig, "fig7b_dense_pair", rows,
            ["reader", "split", "problem_id", "chunk_idx", "accuracy_A", "acc_A_ci_lo", "acc_A_ci_hi",
             "accuracy_B", "acc_B_ci_lo", "acc_B_ci_hi", "cf_importance_A", "cf_importance_B",
             "n_answered_A", "n_answered_B", "n_dissimilar_answered_B", "role", "reader_own_top3", "convergence_idx"])


# fig11_anchor_rank_in_reader_profile removed 2026-09-11 at a request (not needed now or later).

FIGS = {"fig0a": fig0a_sampling_cap, "fig0b": fig0b_prefill_vs_own, "fig3": fig3_screen, "fig3e": fig3e_screen_selected, "fig3d": fig3d_two_traces,
        "fig2c": fig2c_anchor_pairs_by_trace, "fig7b": fig7b_dense_pair, "fig2d": fig2d_pair_examples}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", choices=list(FIGS))
    ap.add_argument("--suffix", default="", help="run-id suffix, e.g. _n32 (figures become fig1*)")
    a = ap.parse_args()
    if a.suffix:
        RUN_SUFFIX, FIG_TAG, N_LABEL = a.suffix, "1", a.suffix.lstrip("_n") or "?"
    PLOTS.mkdir(exist_ok=True)
    for k, fn in FIGS.items():
        if a.only in (None, k):
            fn()
