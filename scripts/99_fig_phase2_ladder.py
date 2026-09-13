#!/usr/bin/env python
"""Phase 2's headline figure: how deep the prefix is cut.

    .venv/bin/python scripts/99_fig_phase2_ladder.py

Reads the three cut arms runs/p2{f,g,c}_<model>/rows.jsonl. Writes
plots/fig98a_phase2_ladder_only.png + _data.csv and prints every plotted value
with its interval and n. No GPU, no server.

A run is drawn only if it is complete: rows on disk == _run.n_cells x
_run.n_per_cell, every cell present, no cell short. Incomplete runs are named
on stdout and left out.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.io import REPO_ROOT, load_config, read_rows, rng
from src.common.stats import cluster_bootstrap

# 3000 by an instruction for this figure; configs/phase2.json ci.resamples
# is 1000, which is untouched. B only sets the Monte-Carlo noise of the percentile
# ends, not the estimate.
RESAMPLES = 3000

# The three cut arms. Both narrower arms are subsets of p2f, and p2c is a subset of p2g
# except for pn 877, which a manual review dropped from p2g only (the lexical rule fired 17
# sentences after the statement, so no margin cleaned it): 15 of p2c's 16 are also in p2g.
ARMS = [("p2f", "answer may be stated", "o"),
        ("p2g", "answer removed", "s"),
        ("p2c", "options never named", "^")]

# Primary first. `sub` = the visually subordinate styling for the self-transplant.
MODELS = [("gpt-oss-20b", "GPT-OSS-20B, a different family (the cross-model case)", False),
          ("r1-distill-14b", "R1-Distill-Qwen-14B, the self-transplant", True)]


def signed(ax, axis):
    """+10 / 0 / −10 on a difference axis, so no tick reads as a level."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:+.0f}".replace("+0", "0").replace("-", "\u2212"))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def figs():
    """scripts/90_make_figures.py as a module: _style, _finish and the colours."""
    spec = importlib.util.spec_from_file_location("figs", REPO_ROOT / "scripts/90_make_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_run(run_id):
    """(per-problem table, n_answered per cell, completeness note or None).

    The table is the metric's first step: one number per (problem, arm, cut) =
    the mean of is_cue over that cell's ANSWERED rollouts (is_cue null = no
    parseable answer, dropped). The stored is_cue column is used as-is; it was
    checked against a full re-extraction with src.common.mcq.extract_mcq
    (scripts/13_analyze_transplant.py:load) on 2026-09-10 and is identical.
    """
    d = REPO_ROOT / "runs" / run_id
    if not (d / "rows.jsonl").exists():
        return None, None, "no rows.jsonl"
    df = pd.DataFrame(list(read_rows(d / "rows.jsonl")))
    r = load_config(d / "config.json")["_run"]
    size = df.groupby(["pn", "arm", "fraction"]).size()
    note = None
    if len(df) != r["n_cells"] * r["n_per_cell"] or len(size) < r["n_cells"] or size.lt(r["n_per_cell"]).any():
        note = (f"{len(df)}/{r['n_cells'] * r['n_per_cell']} rows, {len(size)}/{r['n_cells']} cells, "
                f"{int(size.lt(r['n_per_cell']).sum())} cells under {r['n_per_cell']} samples")
    ans = df[df.is_cue.notna()]
    bad = (ans.is_cue != (ans.answer == ans.cue_answer)).sum()
    assert bad == 0, f"{run_id}: {bad} rows where is_cue disagrees with answer == cue_answer"
    piv = ans.pivot_table(index="pn", columns=["arm", "fraction"], values="is_cue", aggfunc="mean")
    return piv, ans.groupby(["pn", "arm", "fraction"]).size(), note


def effect(piv, fraction, pns=None):
    """Per-problem nudged(fraction) − none, the quantity averaged over problems."""
    d = (piv[("nudged", fraction)] - piv[("none", 0.0)]).dropna()
    return d if pns is None else d.loc[[p for p in d.index if p in set(pns)]]


def stats(d, model, arm, fraction, cat="all"):
    """Mean effect in percentage points with a 95% cluster bootstrap over problems.
    One diff per problem, so the clusters are the problems themselves. The seed
    names the estimand, so the one cell that appears in both panels (p2c at the
    deepest cut) gets the same resample stream in both."""
    salt = (model, arm, fraction, cat)
    s = cluster_bootstrap({pn: [100 * v] for pn, v in d.items()}, RESAMPLES,
                          rng(20260908, "fig98_phase2_ladder", *salt).getrandbits(31))
    return {"value": s["mean_d"], "lo": s["ci_bootstrap"][0], "hi": s["ci_bootstrap"][1],
            "n": s["n_pairs"], "n_positive": round(s["frac_pairs_positive"] * s["n_pairs"]),
            "t_lo": s["ci_t_on_cluster_means"][0], "t_hi": s["ci_t_on_cluster_means"][1]}


def collect(runs):
    """Every plotted point: the ladder's 4 cuts x arm x model."""
    out = []
    for model, _, sub in MODELS:
        for arm, alabel, _ in ARMS:
            piv, n_ans, _ = runs.get((arm, model), (None, None, None))
            if piv is None:
                continue
            for f in (0.25, 0.5, 0.75, 1.0):
                d = effect(piv, f)
                n_roll = int(n_ans.loc[list(d.index), "nudged", f].sum() + n_ans.loc[list(d.index), "none", 0.0].sum())
                out.append({"panel": "ladder", "series": alabel, "model": model, "x": f, "sub": sub,
                            "n_rollouts": n_roll, "baseline": 100 * piv[("none", 0.0)].loc[list(d.index)].mean(),
                            **stats(d, model, arm, f)})
    return out


def draw_ladder(ax, pts, F, colour):
    """Left: the effect against how much of the available prefix was handed over."""
    ax.axhline(0, color=F.INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    for model, _, sub in MODELS:
        for _, alabel, marker in ARMS:
            g = sorted([p for p in pts if p["panel"] == "ladder" and p["model"] == model
                        and p["series"] == alabel], key=lambda p: p["x"])
            if not g:
                continue
            c = colour[alabel]
            ax.errorbar([p["x"] for p in g], [p["value"] for p in g],
                        yerr=[[p["value"] - p["lo"] for p in g], [p["hi"] - p["value"] for p in g]],
                        fmt="none", ecolor=c, elinewidth=1.6 if not sub else 1.0, alpha=0.9 if not sub else 0.75,
                        capsize=0, zorder=3)
            ax.plot([p["x"] for p in g], [p["value"] for p in g], color=c, lw=1.7 if not sub else 0.9,
                    ls="-" if not sub else (0, (3.5, 2)), zorder=4)
            ax.scatter([p["x"] for p in g], [p["value"] for p in g], marker=marker, s=52 if not sub else 40,
                       facecolor=c if not sub else F.SURFACE, edgecolor=F.SURFACE if not sub else c,
                       linewidth=1.3, zorder=5,
                       label=f"{alabel}   ({g[0]['n']} problems)" if not sub else None)
            if not sub:                      # the subordinate set stays unlabelled: it is an anchor, not the result
                last = g[-1]
                ax.annotate(f"{last['value']:+.1f}", (last["x"], last["value"]), textcoords="offset points",
                            xytext=(7, 6), fontsize=8, color=c, va="bottom", fontweight="semibold")
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["25%", "50%", "75%", "100%"])
    ax.set_xlim(0.16, 1.19)
    ax.set_xlabel("share of the available nudged CoT handed to the reader\n(100% = cut at that arm's deepest point)")
    ax.set_ylabel("change in P(hinted option) vs no nudged CoT")
    ax.margins(y=0.12)
    signed(ax, "y")
    ax.set_title("Deeper cuts move the reader further", loc="left", fontweight="bold", fontsize=9.5)
    ax.grid(axis="y", color=F.GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], color=F.MUTED, lw=1.7, marker="o", ms=6, mfc=F.MUTED, mec=F.SURFACE, label=MODELS[0][1]),
          plt.Line2D([], [], color=F.MUTED, lw=0.9, ls=(0, (3.5, 2)), marker="o", ms=5.5, mfc=F.SURFACE,
                     mec=F.MUTED, label=MODELS[1][1])]
    l += [MODELS[0][1], MODELS[1][1]]
    ax.legend(h, l, frameon=False, fontsize=8, loc="upper left", handletextpad=0.6, labelspacing=0.5)


def fig98a_ladder_only(pts, excluded, F):
    """Figure 98a. The effect against how much of A's
    nudged chain of thought the reader was given, for the three cut arms. Metric per
    problem: the mean of is_cue (the reader's final answer == the hinted wrong option)
    over that cell's answered rollouts (50 sampled per cell), plotted as the mean over
    problems of nudged(cut) − none, in percentage points; "none" is the same problem
    asked with no transplanted reasoning. Zero is "the transplant changed nothing", not
    "the reader was right" - the no-transplant baseline sits near the 25% chance rate of
    a 4-option question. The arms differ only in where the CoT is cut, and are drawn from the
    same 58 source problems: p2g and p2c are both subsets of p2f, and 15 of p2c's 16 problems
    are also in p2g (pn 877 was dropped from p2g alone by manual review). Filled/solid: GPT-OSS-20B, a different model
    family. Open/dashed: R1-Distill-Qwen-14B reading its own reasoning back. Intervals
    are 95% cluster bootstraps over problems (3000 resamples)."""
    F._style()
    hue = [F.ANCHOR, F.PARTNER, F.COND["_rec32k"][1]]
    colour = {k: h for (a, lab, _), h in zip(ARMS, hue) for k in (a, lab)}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.17)
    draw_ladder(ax, pts, F, colour)
    ax.set_title("", loc="left")                 # the headline is the figure's, not the panel's
    if not any(p["panel"] == "ladder" for p in pts):
        raise SystemExit("no complete Phase 2 run found under runs/p2{f,g,c}_<model>/. Generate them with "
                         "scripts/12_run_transplant.py, or unpack the data bundle's rollouts/p2* directories "
                         "into runs/ - see the README. The survey above lists what was found.")
    base = next(p for p in pts if p["panel"] == "ladder" and p["model"] == MODELS[0][0]
                and p["series"] == "answer may be stated")["baseline"]
    fig.suptitle("The deeper the nudged CoT is cut, the further the reader moves",
                 x=0.015, ha="left", fontsize=11.5, fontweight="bold", y=1.05)
    fig.text(0.015, 1.005,
             f"means over problems, 95% cluster-bootstrap intervals over problems ({RESAMPLES} resamples)"
             + (f"\nnot drawn, run incomplete: {', '.join(excluded)}" if excluded else ""),
             fontsize=8.6, color=F.MUTED, va="top", linespacing=1.5)
    rows = [[p["series"], p["model"], "self-transplant" if p["sub"] else "primary", p["x"],
             round(p["value"], 3), round(p["lo"], 3), round(p["hi"], 3), p["n"], p["n_positive"],
             p["n_rollouts"], round(p["baseline"], 3),
             f"t-CI on problem means [{p['t_lo']:+.3f}, {p['t_hi']:+.3f}]"]
            for p in pts if p["panel"] == "ladder"]
    F._finish(fig, "fig98a_phase2_ladder_only", rows,
              ["series", "model", "role", "x_fraction_of_cot", "effect_pp", "ci_lo_pp", "ci_hi_pp",
               "n_problems", "n_problems_positive", "n_answered_rollouts", "baseline_none_pct", "note"])


def survey():
    """Every p2{f,g,c}_<model> run on disk, checked for completeness. Returns the
    complete ones the figure draws, plus a line per run for stdout: a model is
    drawn only if it is in MODELS and every arm it contributes is complete."""
    runs, excluded, lines = {}, [], []
    drawn = [m for m, _, _ in MODELS]
    for d in sorted((REPO_ROOT / "runs").glob("p2[fgc]_*")):
        arm, model = d.name.split("_", 1)
        piv, n_ans, note = load_run(d.name)
        if note:
            excluded.append(d.name)
            lines.append(f"EXCLUDED    {d.name:26s} incomplete: {note}")
        elif model in drawn:
            runs[(arm, model)] = (piv, n_ans, note)
            lines.append(f"DRAWN       {d.name:26s} complete")
        else:
            lines.append(f"NOT DRAWN   {d.name:26s} complete, but {model} is not a figure model")
    return runs, excluded, lines


def main():
    F = figs()
    runs, excluded, lines = survey()
    pts = collect(runs)
    fig98a_ladder_only(pts, excluded, F)
    print(f"{'panel':8s} {'series':32s} {'model':16s} {'cut':>5s} {'effect_pp':>10s} "
          f"{'95% CI (pp)':>20s} {'n_prob':>7s} {'n>0':>4s} {'rollouts':>9s}")
    for p in pts:
        ci = "[%+.1f, %+.1f]" % (p["lo"], p["hi"])
        print(f"{p['panel']:8s} {p['series']:32s} {p['model']:16s} {p['x']:5.2f} {p['value']:+10.1f} "
              f"{ci:>20s} {p['n']:7d} {p['n_positive']:4d} {p['n_rollouts']:9d}")
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
