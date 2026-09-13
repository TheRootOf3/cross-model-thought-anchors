#!/usr/bin/env python
"""Export every Phase 2 selection and classification decision for manual review.

    python scripts/15_export_decisions.py [--sample 8] [--seed 20260908] [--out_dir doc/decisions]

Writes two files:
  decisions.csv one row per unique problem (58), with every decision and the exact
                  sentence that triggered it - so each can be checked without rerunning anything
  SAMPLE.md a seeded random handful, laid out in full: the sentence the rule fired on,
                  the last sentence kept, the first sentence cut, and the blind raters' labels

Three kinds of decision are recorded, each traceable to a sentence:
  SET 1 cut the "commit" rule: first sentence committing to the HINTED option
  SET 2 cut the "any_option" rule: first sentence picking out ANY option (symmetric)
  regime label STATES / ARGUES / NEITHER for the hinted option in the SET 1 deepest prefix,
                  from three independent raters who were blind to every experimental outcome
                  (data/phase2_prefix_labels.json), disagreements adjudicated by a fourth
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import REPO_ROOT, load_config, rng, run_dir
from src.common.split import split_solution_into_chunks

_spec = importlib.util.spec_from_file_location("sel11", Path(__file__).parent / "11_select_traces_p2.py")
sel11 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sel11)


def why_any_option(sentences, dist, opts, min_words):
    """Re-derive the SET 2 trigger and say WHICH option it matched and HOW."""
    import re
    thr = {k: min(min_words, len(v)) or 99 for k, v in dist.items()}
    for i, s in enumerate(sentences):
        w = set(re.findall(r"[a-z0-9']+", s.lower()))
        letters = {k for k in dist if sel11.letter_ref(s, k)}
        content = {k: sorted(dist[k] & w) for k in dist if len(dist[k] & w) >= thr[k]}
        if (letters or content) and (sel11.ENDORSE.search(s) or content):
            bits = []
            if letters: bits.append("names option " + "/".join(sorted(letters)) + " by letter")
            for k, ws in content.items(): bits.append(f"uses option {k}'s words {ws}")
            if sel11.ENDORSE.search(s): bits.append(f"endorsement phrase {sel11.ENDORSE.search(s).group(0)!r}")
            return i, s, "; ".join(bits)
    return None, None, "no sentence picks out any option"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out_dir", default="doc/decisions")
    args = ap.parse_args()
    cfg = load_config("configs/phase2.json")
    mw = cfg["selection"]["content_min_distinctive_words"]
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    _s1 = json.load(open(REPO_ROOT / "data/phase2_problems.json"))
    set1 = {p["pn"]: p for p in _s1["problems"]}
    # the drops come from the SELECTION OUTPUT, not runs/<id>/config.json: write_run_config only writes
    # when the file is absent (io.py:33), so a rerun under the same run_id leaves a stale config behind.
    drops = {d["pn"]: d for d in _s1["dropped"]}
    set2 = {p["pn"]: p for p in json.load(open(REPO_ROOT / "data/phase2_problems_strict.json"))["problems"]}
    lab = json.load(open(REPO_ROOT / "data/phase2_prefix_labels.json"))

    rows = []
    universe = sorted(set(set1) | set(drops))
    for pn in universe:
        p = set1.get(pn)
        d = drops.get(pn)
        r = {"pn": pn, "in_set1": pn in set1, "set1_drop_reason": (d or {}).get("reason", ""),
             "set1_drop_detail": (d or {}).get("detail", "")[:200]}
        if p:
            sents = split_solution_into_chunks(p["reasoning_text"])
            cue = p["cue_answer"]
            j = lab["judgements"].get(f"{pn}:{cue}", {})
            final = j.get("final")
            r |= {"n_sentences": p["n_sentences"], "hinted_option": cue,
                  "hinted_option_text": p["options"][cue][:80], "gt_option": p["gt_answer"],
                  "set1_cut_sentences": p["last_safe_sentence"],
                  "set1_trigger_idx": p["leak_idx"], "set1_trigger_rule": p["leak_detector"] or "none found",
                  "set1_trigger_sentence": (p["leak_sentence"] or "")[:300],
                  "blind_label_hinted_option": final, "blind_label_agreement": j.get("how", ""),
                  "blind_label_raters": "/".join(j.get("raters", {}).values()),
                  "regime": {"STATES": "1 states", "ARGUES": "2 argues"}.get(final, "3 neither"),
                  "adjudicator_quote": j.get("adjudicator_quote", "")[:200]}
            dist = sel11.distinctive(p["options"])
            i2, s2, why = why_any_option(sents, dist, p["options"], mw)
            r |= {"set2_trigger_idx": i2, "set2_trigger_sentence": (s2 or "")[:300], "set2_trigger_why": why[:240],
                  "in_set2_min4": pn in set2, "set2_cut_sentences": set2.get(pn, {}).get("last_safe_sentence", "")}
        rows.append(r)

    cols = ["pn", "in_set1", "set1_drop_reason", "set1_drop_detail", "n_sentences", "hinted_option",
            "hinted_option_text", "gt_option", "set1_cut_sentences", "set1_trigger_idx", "set1_trigger_rule",
            "set1_trigger_sentence", "blind_label_hinted_option", "blind_label_agreement", "blind_label_raters",
            "regime", "adjudicator_quote", "set2_trigger_idx", "set2_trigger_sentence", "set2_trigger_why",
            "in_set2_min4", "set2_cut_sentences"]
    with open(out / "decisions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # --- the sample: spread over decision types so the author checks each kind once ----------
    r_ = rng(args.seed, "p2_review_sample")
    buckets = {
        "SET 1 DROPPED": [x for x in rows if not x["in_set1"]],
        "regime 1 STATES": [x for x in rows if x.get("regime") == "1 states"],
        "regime 2 ARGUES": [x for x in rows if x.get("regime") == "2 argues"],
        "regime 3 NEITHER": [x for x in rows if x.get("regime") == "3 neither"],
        "in SET 2 (strict cut kept it)": [x for x in rows if x.get("in_set2_min4")],
        "SET 2 dropped (no neutral prefix)": [x for x in rows if x["in_set1"] and not x.get("in_set2_min4")],
    }
    per = max(1, args.sample // len(buckets))
    md = [f"# Phase 2 decisions - review sample\n",
          f"Seed {args.seed}. {args.sample}-ish problems, spread over the kinds of decision so each kind gets "
          f"checked once. Full table: `decisions.csv` ({len(rows)} problems).\n",
          "For each: the sentence a rule fired on, the last sentence KEPT and the first sentence CUT under "
          "both cut rules, and the three blind raters' label for the hinted option. "
          "**What to check:** does the first CUT sentence justify cutting there, and does the last KEPT "
          "sentence avoid giving the answer away?\n"]
    seen = set()
    for name, pool in buckets.items():
        pool = [x for x in pool if x["pn"] not in seen]
        pick = r_.sample(pool, min(per, len(pool)))
        md.append(f"\n---\n\n## {name}  ({len(pool)} available, showing {len(pick)})\n")
        for x in pick:
            seen.add(x["pn"])
            md.append(f"### pn {x['pn']}")
            if not x["in_set1"]:
                md.append(f"- **DROPPED from Set 1**: {x['set1_drop_reason']} - {x['set1_drop_detail']}\n")
                continue
            p = set1[x["pn"]]; sents = split_solution_into_chunks(p["reasoning_text"])
            md.append(f"- hinted (WRONG) option **({x['hinted_option']})** \"{x['hinted_option_text']}\"; "
                      f"ground truth ({x['gt_option']}). {x['n_sentences']} sentences of reasoning.")
            md.append(f"- blind raters on the hinted option: **{x['blind_label_hinted_option']}** "
                      f"({x['blind_label_agreement']}; individually {x['blind_label_raters']}) -> regime *{x['regime']}*")
            if x["adjudicator_quote"]:
                md.append(f"- adjudicator: {x['adjudicator_quote']}")
            md.append(f"\n**Set 1 cut** - rule fired at sentence {x['set1_trigger_idx']} "
                      f"({x['set1_trigger_rule']}), keeping {x['set1_cut_sentences']}:")
            k = x["set1_cut_sentences"]
            md.append(f"  - last KEPT  #{k-1}: {sents[k-1][:260] if k >= 1 else '(none)'}")
            md.append(f"  - first CUT  #{k}: {sents[k][:260] if k < len(sents) else '(end of trace)'}")
            md.append(f"\n**Set 2 cut** - first sentence picking out any option is #{x['set2_trigger_idx']}; "
                      f"why: {x['set2_trigger_why']}")
            if x["in_set2_min4"]:
                k2 = x["set2_cut_sentences"]
                md.append(f"  - kept {k2} sentences. last KEPT #{k2-1}: {sents[k2-1][:260]}")
                md.append(f"  - first CUT #{k2}: {sents[k2][:260]}")
            else:
                md.append(f"  - **not in Set 2**: fewer than {cfg['selection']['min_prefix_sentences']}+ option-free sentences")
            md.append("")
    (out / "SAMPLE.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} problems -> {args.out_dir}/decisions.csv")
    print(f"{len(seen)} sampled -> {args.out_dir}/SAMPLE.md")
    for name, pool in buckets.items():
        print(f"    {name:<36} {len(pool):>3} problems")


if __name__ == "__main__":
    main()
