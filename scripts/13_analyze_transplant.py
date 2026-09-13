#!/usr/bin/env python
"""Phase 2 readout: does a hidden nudge in A's reasoning move a different model?

    python scripts/13_analyze_transplant.py [--models r1-distill-14b gpt-oss-20b qwen3-1.7b]
        [--run_prefix p2_] [--forced] [--out doc/p2_results.md] [--fig fig20_nudge_transfer]

Reads runs/<prefix><model>/rows.jsonl and RE-EXTRACTS every answer from the
stored completion text, so the extraction rule can be revised without
regenerating a single rollout (src/common/mcq.py). Nothing here trusts the
`answer` column written at generation time; it is recomputed and the two are
compared, and any disagreement is reported.

Primary metric, thought-branches': p_cue = P(final answer == the hinted WRONG
answer | an answer was parsed), 4 options so chance is 0.25. Effects are
cluster-bootstrapped over PROBLEMS, which is the unit - 45 of them, deduplicated
from the 71 rows the released file ships.

Two contrasts, because they answer different questions:
  nudged - none against no reasoning at all. The neutral comparison.
  nudged - clean against a length-matched prefix of A's UNCUED, correct
                   reasoning. Larger, because the clean arm actively pushes
                   toward the truth; report both and say so.

The pre-registered decision rule (configs/phase2.json) is checked first and
printed literally: if A does not replicate, the readers are not interpreted.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import REPO_ROOT, load_config, read_rows, rng, run_dir
from src.common.mcq import extract_mcq


def load(model: str, run_id: str, forced: bool) -> pd.DataFrame:
    path = run_dir(run_id, create=False) / ("forced.jsonl" if forced else "rows.jsonl")
    rows = list(read_rows(path))
    assert rows, f"no rows at {path}"
    df = pd.DataFrame(rows)
    bad = df.groupby(["pn", "arm", "fraction"]).prompt_sha.nunique().max()
    assert bad == 1, (f"{run_id}: {bad} different prompts inside one cell - the cut moved and the run was "
                      f"resumed with the same run_id, so rows from two stimuli would be averaged. Use a new run_id.")
    cfg_path = run_dir(run_id, create=False) / "config.json"
    if cfg_path.exists():
        r = json.load(open(cfg_path))["_run"]
        cells_have = df.groupby(["pn", "arm", "fraction"]).ngroups
        short = df.groupby(["pn", "arm", "fraction"]).size().lt(r["n_per_cell"]).sum()
        if cells_have < r["n_cells"] or short:
            df.attrs["incomplete"] = f"{cells_have}/{r['n_cells']} cells, {short} of them under {r['n_per_cell']} samples"
    if not forced:                       # recompute from the raw text; never trust the stored column
        redo = [extract_mcq(t, model, f) for t, f in zip(df.text, df.finish_reason)]
        df["answer_recomputed"] = [a for a, _ in redo]
        df["tier"] = [t for _, t in redo]
        df["is_cue"] = np.where(df.answer_recomputed.isna(), np.nan, df.answer_recomputed == df.cue_answer)
        df["is_gt"] = np.where(df.answer_recomputed.isna(), np.nan, df.answer_recomputed == df.gt_answer)
    return df


def per_problem(df: pd.DataFrame, col: str = "is_cue") -> pd.DataFrame:
    """One number per (problem, arm, cut): the rate over that cell's answered rollouts."""
    d = df[df[col].notna()]
    return d.pivot_table(index="pn", columns=["arm", "fraction"], values=col, aggfunc="mean")


def boot(diff: pd.Series, resamples: int, level: float, seed: int) -> tuple[float, float]:
    r = rng(seed, "p2_bootstrap")
    v = diff.dropna().values
    means = [np.mean([v[r.randrange(len(v))] for _ in range(len(v))]) for _ in range(resamples)]
    a = (1 - level) / 2
    return float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a)))


def contrasts(piv: pd.DataFrame, fractions: list[float], cfg: dict) -> list[dict]:
    ci, out = cfg["ci"], []
    for f in fractions:
        for base in ("none", "clean"):
            b = piv[("none", 0.0)] if base == "none" else piv[("clean", f)]
            d = (piv[("nudged", f)] - b).dropna()
            lo, hi = boot(d, ci["resamples"], ci["level"], 20260908)
            out.append({"fraction": f, "against": base, "nudged": piv[("nudged", f)].mean(),
                        "baseline": b.mean(), "diff": d.mean(), "lo": lo, "hi": hi,
                        "n_problems": len(d), "n_positive": int((d > 0).sum())})
    return out


def figure(rows: pd.DataFrame, name: str):
    """Figure 20. A hidden nudge in one model's chain of thought moves a different model.

    P(the hinted, wrong option) against how much of the nudged reasoning the model
    was given, for the model that wrote it (self) and for two models that did not.
    Both control arms are shown: no reasoning at all, and a length-matched prefix
    of the same author's UNCUED, correct reasoning. Points are means over
    problems with 95% cluster-bootstrap intervals over problems; the dashed line
    is chance for a 4-option question. Unit: probability of a final answer."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from importlib import import_module
    mk = import_module("90_make_figures")
    mk._style()
    models = list(dict.fromkeys(rows.model))
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    style = {"nudged": (mk.ANCHOR, "o", "nudged prefix"), "clean": (mk.PARTNER, "s", "clean prefix"),
             "none": (mk.INK, "^", "no prefix")}
    for ax, m in zip(axes, models):
        g = rows[rows.model == m]
        for arm, (c, mk_, lab) in style.items():
            h = g[g.arm == arm].sort_values("fraction")
            if arm == "none":
                ax.axhline(h.p_cue.iloc[0], color=c, lw=1, ls=(0, (1, 2)), zorder=2)
                ax.plot([], [], color=c, marker=mk_, ls="none", label=lab)
                continue
            ax.errorbar(h.fraction, h.p_cue, yerr=[h.p_cue - h.lo_cue, h.hi_cue - h.p_cue],
                        color=c, marker=mk_, ms=5, lw=1.4, capsize=2.5, zorder=3, label=lab)
        ax.axhline(0.25, color="#999", lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_title(f"{m}\n({g.role.iloc[0]})", fontsize=9)
        ax.set_xlabel("fraction of the nudged reasoning given")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("P(the hinted, wrong option)")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("A hidden nudge survives being read by a different model", x=0.02, ha="left",
                 fontsize=10, fontweight="bold")
    mk._finish(fig, name, rows.round(4).values.tolist(), list(rows.columns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["r1-distill-14b", "gpt-oss-20b", "qwen3-1.7b"])
    ap.add_argument("--run_prefix", default="p2_")
    ap.add_argument("--config", default="configs/phase2.json")
    ap.add_argument("--forced", action="store_true", help="read the forced-answer probe instead")
    ap.add_argument("--out", default="doc/p2_results.md")
    ap.add_argument("--fig", default="fig20_nudge_transfer")
    args = ap.parse_args()
    cfg = load_config(args.config)
    roles = {"r1-distill-14b": "self: wrote the reasoning", "gpt-oss-20b": "far: different family",
             "qwen3-1.7b": "near: same family, 1.7B"}

    md, plot_rows, verdict = [], [], {}
    md.append(f"# Phase 2: nudge transfer{' (FORCED-ANSWER PROBE)' if args.forced else ''}\n")
    md.append(f"Metric: P(final answer == the hinted wrong option), answered-only. 4 options, chance 0.25. "
              f"Intervals are {int(cfg['ci']['level']*100)}% cluster bootstraps over problems "
              f"({cfg['ci']['resamples']} resamples).\n")
    for m in args.models:
        df = load(m, f"{args.run_prefix}{m}", args.forced)
        inc = df.attrs.get("incomplete")
        # the grid the run was GENERATED on, not the one the config happens to hold now
        fractions = sorted(f for f in df.fraction.unique() if f > 0)
        piv_cue, piv_gt = per_problem(df, "is_cue"), per_problem(df, "is_gt")
        parsed = df.is_cue.notna().mean()
        if not args.forced:
            mismatch = (df.answer.astype(object) != df.answer_recomputed.astype(object)).sum()
            md.append((f"\n> **INCOMPLETE RUN: {inc}.** Nothing below is a result.\n" if inc else "")
                      + f"\n## {m} ({roles.get(m,'')}) - {len(df)} rollouts, {parsed:.1%} parsed, "
                      f"{mismatch} answers changed on re-extraction, tiers "
                      f"{dict(df.tier.value_counts(dropna=False))}\n")
        else:
            md.append(f"\n## {m} ({roles.get(m,'')}) - {len(df)} probes, {parsed:.1%} parsed\n")
        md.append("| cut | none | clean | nudged | nudged-none | nudged-clean | problems + (vs none) |")
        md.append("|---|---|---|---|---|---|---|")
        cs = contrasts(piv_cue, fractions, cfg)
        for f in fractions:
            vn = next(c for c in cs if c["fraction"] == f and c["against"] == "none")
            vc = next(c for c in cs if c["fraction"] == f and c["against"] == "clean")
            md.append(f"| {f} | {piv_cue[('none',0.0)].mean():.3f} | {piv_cue[('clean',f)].mean():.3f} | "
                      f"{vn['nudged']:.3f} | {vn['diff']:+.3f} [{vn['lo']:+.3f}, {vn['hi']:+.3f}] | "
                      f"{vc['diff']:+.3f} [{vc['lo']:+.3f}, {vc['hi']:+.3f}] | {vn['n_positive']}/{vn['n_problems']} |")
            for arm in ("nudged", "clean"):
                lo, hi = boot(piv_cue[(arm, f)], cfg["ci"]["resamples"], cfg["ci"]["level"], 20260908)
                plot_rows.append({"model": m, "role": roles.get(m, ""), "arm": arm, "fraction": f,
                                  "p_cue": piv_cue[(arm, f)].mean(), "lo_cue": lo, "hi_cue": hi,
                                  "p_gt": piv_gt[(arm, f)].mean(), "n_problems": piv_cue[(arm, f)].notna().sum()})
        lo, hi = boot(piv_cue[("none", 0.0)], cfg["ci"]["resamples"], cfg["ci"]["level"], 20260908)
        plot_rows.append({"model": m, "role": roles.get(m, ""), "arm": "none", "fraction": 0.0,
                          "p_cue": piv_cue[("none", 0.0)].mean(), "lo_cue": lo, "hi_cue": hi,
                          "p_gt": piv_gt[("none", 0.0)].mean(), "n_problems": piv_cue[("none", 0.0)].notna().sum()})
        deepest = next(c for c in cs if c["fraction"] == max(fractions) and c["against"] == "clean")
        deepest["fractions"] = fractions
        verdict[m] = deepest
        md.append(f"\nP(ground truth) at the deepest cut: nudged {piv_gt[('nudged',max(fractions))].mean():.3f}, "
                  f"clean {piv_gt[('clean',max(fractions))].mean():.3f}, none {piv_gt[('none',0.0)].mean():.3f}.")

    a = verdict.get("r1-distill-14b")
    if a is None:
        md.insert(2, "\n**DECISION RULE NOT EVALUATED** - the replication arm (r1-distill-14b) was not in "
                     "--models, so nothing below is licensed as a result.\n")
    else:
        ok = a["lo"] > 0
        md.insert(2, f"\n**DECISION RULE** (configs/phase2.json, fixed before any Phase 2 rollout): "
                     f"*{cfg['decision_rule']}*\n\n**VERDICT: {'REPLICATION HOLDS' if ok else 'FAILED REPLICATION'}** - "
                     f"A gives {a['diff']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}] over {a['n_problems']} problems"
                     + ("" if ok else ". The readers' numbers below are NOT interpreted.") + "\n")
    (REPO_ROOT / args.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    rows = pd.DataFrame(plot_rows)
    if not args.forced:
        figure(rows, args.fig)
    rows.round(4).to_csv(REPO_ROOT / args.out.replace(".md", "_data.csv"), index=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
