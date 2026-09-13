#!/usr/bin/env python
"""Phase 2 trace selection: dedupe, re-screen, find the leak, cut before it.

    python scripts/11_select_traces_p2.py [--config configs/phase2.json]
        [--out data/phase2_problems.json] [--run_id h11_select_p2] [--show 8] [--ack]

thought-branches ships 71 "good problems" - chains of thought written by
R1-Distill-Qwen-14B under a hidden "a Stanford Professor answered (X)" hint,
which reach the hinted wrong answer without mentioning the hint. Three things
make them unusable as shipped (2026-09-10):

  * 71 rows are 58 unique problems (12 duplicate groups, @pkld cache).
  * its no-mention filter is four lowercase substrings plus a judge function
    that is undefined in the repo, and it misses reasoning that says
    "the original answer was (C)" - a hint leak in plain sight.
  * `answer`, `original_answer`, `cond` and `model` describe the Chua CSV's
    model, not the one that wrote the reasoning.

So we dedupe, re-screen, and locate the LEAK - the first sentence that commits
to the hinted answer - by three detectors, taking whichever fires first:
  meta the sentence refers to an externally given answer (should be empty
           after screening; if it fires the problem is dropped)
  letter it names the hinted option
  content it uses words distinctive to the hinted option and to no other
The prefix is cut `leak_safety_margin_sentences` before that, because a
paraphrase of the option often precedes the letter by one sentence.

CHECKPOINT: prints every selected problem's cut, and the full text of
--show of them, then exits 2 unless --ack. Read the ones flagged NO LEAK first:
there the whole chain of thought is handed over, so if the rule missed a
commitment the experiment is confounded.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import REPO_ROOT, load_config, write_run_config
from src.common.split import split_solution_into_chunks
from src.phase2.chunk_ranges import get_chunk_ranges

STOP = set("the a an of to in on for and or is are was were be been by with as at that this it its "
           "which who whom whose from not no nor but if then than so such can could would should may "
           "might must do does did have has had they them their there these those he she his her".split())


def parse_options(question: str) -> dict[str, str]:
    """The four option strings, from the question text itself - there is no options field."""
    block = question.split("Answer choices:")[-1].split("Please think step by step")[0]
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\((\w)\)\s*(.+)$", block, re.M)}


def distinctive(options: dict[str, str]) -> dict[str, set[str]]:
    """Words that appear in one option and in no other; the basis of the content detector.
    Digits count: several option sets differ only in a number ('At least 13' vs 'At least 20')."""
    words = {k: {w for w in re.findall(r"[a-z0-9']+", v.lower()) if w not in STOP and len(w) > 1}
             for k, v in options.items()}
    return {k: v - set().union(*(w for j, w in words.items() if j != k)) for k, v in words.items()}


# A sentence only leaks if it COMMITS. Describing an option ("Option B is about X") does not
# tell the reader what the answer is, and cutting there throws away most of the prefix for nothing.
# Rule "any_option" (2026-09-10, after a check found the commit rule leaks): cut before the first
# sentence that picks out ANY option - by endorsing it, or simply by naming its distinctive content.
# Symmetric across the four options, so it cannot favour the hinted one, and over-firing only cuts
# earlier. This is the rule to use when the prefix must be neutral ABOUT THE OPTIONS, not merely
# silent about the hinted one; the "commit" rule below was too narrow and its prefixes stated the
# answer in 19 of 45 problems (blind labelling, 2026-09-10).
ENDORSE = re.compile(
    r"\b(?:which|that)\s+(?:is|would\s+be)\b|\bthe\s+only\s+(?:option|one|choice)\s+(?:left|remaining)\b"
    r"|\b(?:leaning|inclined)\s+tow(?:ards?|ard)\b|\bfits\s+best\b|\bbest\s+fit\b|\bmakes?\s+more\s+sense\b"
    r"|\bmost\s+(?:plausible|accurate|logical|likely|consistent)\b|\bodd\s+one\s+out\b|\bweakest\b|\bstrongest\b"
    r"|\b(?:best|correct|right)\s+answer\b|\bgo(?:ing)?\s+with\b|\bmust\s+be\b|\bhas\s+to\s+be\b"
    r"|\btherefore\b|\bthus\b|\bhence\b|\bprobably\b|\bdefinitely\b|\bi'?m\s+(?:pretty\s+|fairly\s+)?(?:sure|confident)\b"
    r"|\b(?:seems?|sounds?|looks?)\s+(?:to\s+be\s+)?(?:like\s+)?(?:the\s+)?(?:best|correct|right|most)\b"
    r"|\bi\s+think\b|\bmakes?\s+(?:option\s+)?[A-D]\b", re.I)


def find_any_option(sentences, dist, meta, min_words):
    """First sentence that picks out any option at all. Returns (index, detector)."""
    thr = {k: min(min_words, len(v)) or 99 for k, v in dist.items()}
    for i, s in enumerate(sentences):
        if meta.search(s.lower()):
            return i, "meta"
        w = set(re.findall(r"[a-z0-9']+", s.lower()))
        refs = {k for k in dist if letter_ref(s, k) or len(dist[k] & w) >= thr[k]}
        if refs and (ENDORSE.search(s) or any(len(dist[k] & w) >= thr[k] for k in refs)):
            return i, "any_option"
    return None, None


COMMIT = re.compile(
    r"\b(?:best |correct |right |final )?answers?\s+(?:is|are|would be|should be|must be|has to be)\b"
    r"|\bi'?(?:ll|m)\s+go(?:ing)?\s+with\b|\bgo\s+with\b|\bthat\s+leaves\b|\bleaves?\s+(?:us\s+)?with\b"
    r"|\bmust\s+be\b|\bhas\s+to\s+be\b|\bso\s+it'?s\b|\btherefore\b|\bthus\b|\bhence\b"
    r"|\b(?:the\s+)?(?:best|correct|right)\s+(?:option|choice|one|answer)\b"
    r"|\b(?:seems?|sounds?|looks?)\s+(?:to\s+be\s+)?(?:like\s+)?(?:the\s+)?(?:best|correct|right)\b"
    r"|\bis\s+(?:the\s+)?(?:most\s+)?(?:likely|probably|definitely|clearly)\b"
    r"|\bi'?m\s+(?:pretty\s+|fairly\s+)?(?:sure|confident)\b|\\boxed\b|\bi\s+think\s+it'?s\b",
    re.I)


def letter_ref(sentence: str, letter: str) -> bool:
    """Does this sentence point at that option by its letter?"""
    return bool(re.search(rf"\(\s*{letter}\s*\)|\boption\s+{letter}\b|\bchoice\s+{letter}\b"
                          rf"|\b{letter}\)|(?<![A-Za-z]){letter}(?![A-Za-z'])", sentence))


def find_leak(sentences: list[str], cue: str, dist: dict[str, set[str]], meta: re.Pattern,
              min_words: int) -> tuple[int | None, str | None]:
    """Index of the first sentence that COMMITS to the hinted option, and which detector saw it.

    Three detectors, first hit wins:
      meta refers to an answer given from outside ("the original answer was (C)")
      letter a commitment phrase in a sentence naming the hinted option and no other
      content a commitment phrase plus words distinctive to the hinted option and no other
    A sentence that merely describes an option, or enumerates several, is not a leak.
    """
    thr = {k: min(min_words, len(v)) or 99 for k, v in dist.items()}
    for i, s in enumerate(sentences):
        low = s.lower()
        if meta.search(low):
            return i, "meta"
        committing = bool(COMMIT.search(s))
        letters = {k for k in dist if letter_ref(s, k)}
        if committing and letters == {cue}:
            return i, "letter"
        if committing:
            w = set(re.findall(r"[a-z0-9']+", low))
            hits = {k for k in dist if len(dist[k] & w) >= thr[k]}
            if hits == {cue} and not letters - {cue}:
                return i, "content"
    return None, None


def cut_offsets(text: str, fractions: list[float], last_sentence: int) -> list[dict]:
    """Character offsets of the cut points: cut k = text[:offset], ending at a sentence boundary."""
    sents = split_solution_into_chunks(text)
    ranges = get_chunk_ranges(text, sents)
    out = []
    for f in fractions:
        k = min(round(f * last_sentence), len(ranges) - 1)   # never past the last sentence START:
        out.append({"fraction": f, "n_prefix_sentences": k,   # `len(text)` handed over the whole trace,
                    "char": ranges[k][0]})                    # a prompt their sweep never contains
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.json")
    ap.add_argument("--out", default="data/phase2_problems.json")
    ap.add_argument("--run_id", default="h11_select_p2")
    ap.add_argument("--show", type=int, default=8, help="print this many full traces for the author")
    ap.add_argument("--rule", default="commit", choices=["commit", "any_option"],
                    help="commit: cut before the reasoning commits to the HINTED option (leaks: 19 of 45 still state it). "
                         "any_option: cut before it picks out ANY option (symmetric, much stricter, much shorter)")
    ap.add_argument("--min_prefix", type=int, default=None, help="override selection.min_prefix_sentences")
    ap.add_argument("--margin", type=int, default=None,
                    help="override selection.leak_safety_margin_sentences. The default 1 left the STATED sentence "
                         "inside the prefix on 9 of 45 problems, because the rule often fires only at the trace's "
                         "last sentence; 4 cleans 8 of those 9 (manual review 2026-09-10)")
    ap.add_argument("--exclude_pn", type=int, nargs="*", default=(),
                    help="drop these problem numbers by hand, with the reason recorded in the output")
    ap.add_argument("--ack", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sel = cfg["selection"]
    meta = re.compile("|".join(sel["cue_verbalisation_patterns"]), re.I)
    raw = json.load(open(REPO_ROOT / cfg["dataset"]))

    # 1. dedupe: 12 groups are byte-identical apart from pn and the (inert) model field
    groups = defaultdict(list)
    for r in raw:
        groups[tuple(r[k] for k in sel["dedupe_on"])].append(r)
    uniq = [(v[0], [x["pn"] for x in v[1:]]) for v in groups.values()]

    kept, dropped = [], []
    for rec, dupes in uniq:
        pn, cue, gt = rec["pn"], rec["cue_answer"], rec["gt_answer"]
        drop = lambda why, detail="": dropped.append({"pn": pn, "reason": why, "detail": detail})
        if pn in args.exclude_pn:
            drop("excluded by hand")
            continue
        opts = parse_options(rec["question"])
        if len(opts) != cfg["n_options"] or cue not in opts or gt not in opts:
            drop("options unparseable", f"parsed {sorted(opts)}")
            continue
        dist = distinctive(opts)
        if not dist[cue]:
            drop("options share every word", f"{opts[cue][:70]!r} has no word the other options lack")
            continue
        sents = split_solution_into_chunks(rec["reasoning_text"])
        idx, det = (find_any_option(sents, dist, meta, sel["content_min_distinctive_words"])
                    if args.rule == "any_option"
                    else find_leak(sents, cue, dist, meta, sel["content_min_distinctive_words"]))
        if det == "meta":
            drop("reasoning verbalises the hint", sents[idx][:150])
            continue
        # never hand over the whole trace: 13 of 14 "no leak found" problems did exactly that
        margin = args.margin if args.margin is not None else sel["leak_safety_margin_sentences"]
        last = len(sents) - 3 if idx is None else idx - margin
        if last < (args.min_prefix if args.min_prefix is not None else sel["min_prefix_sentences"]):
            drop("leak too early to leave a prefix", f"leak at sentence {idx} of {len(sents)}")
            continue
        clean_text = rec["base_gt_reasoning_text"]
        cuts = cut_offsets(rec["reasoning_text"], cfg["cuts"]["fractions"], last)
        clean_sents = split_solution_into_chunks(clean_text)
        clean_ranges = get_chunk_ranges(clean_text, clean_sents)
        clean_cuts = []
        for c in cuts:                       # match the clean arm on prefix length in characters
            j = min(range(len(clean_ranges)), key=lambda k: abs(clean_ranges[k][0] - c["char"])) if c["char"] else 0
            clean_cuts.append({"fraction": c["fraction"], "n_prefix_sentences": j,
                               "char": clean_ranges[j][0] if c["char"] else 0, "chars_wanted": c["char"]})
        kept.append({"pn": pn, "duplicate_pns": dupes, "cue_answer": cue, "gt_answer": gt,
                     "options": opts, "question": rec["question"],
                     "n_sentences": len(sents), "leak_idx": idx, "leak_detector": det,
                     "leak_sentence": None if idx is None else sents[idx],
                     "last_safe_sentence": last, "cuts": cuts,
                     "clean_n_sentences": len(clean_sents), "clean_cuts": clean_cuts,
                     "reasoning_text": rec["reasoning_text"], "clean_reasoning_text": clean_text})

    out = {"config": args.config, "dataset": cfg["dataset"], "n_raw": len(raw), "n_unique": len(uniq),
           "n_selected": len(kept), "dropped": dropped, "problems": kept}
    (REPO_ROOT / args.out).write_text(json.dumps(out, indent=2))
    write_run_config(args.run_id, cfg, extra={"script": "11_select_traces_p2", "args": vars(args),
                                              "n_raw": len(raw), "n_unique": len(uniq), "n_selected": len(kept),
                                              "dropped": dropped})

    print(f"{len(raw)} rows -> {len(uniq)} unique problems -> {len(kept)} selected  ({args.out})")
    print(f"dropped {len(dropped)}: " + ", ".join(f"{k} x{v}" for k, v in Counter(d['reason'] for d in dropped).items()))
    for d in dropped:
        print(f"  pn {d['pn']:<5} {d['reason']:<38} {d['detail'][:90]}")
    print(f"\ndetector that found the leak: " + ", ".join(f"{k or 'NO LEAK'} x{v}" for k, v in
                                                          Counter(p["leak_detector"] for p in kept).items()))
    print(f"\n{'pn':<6}{'sents':>6}{'leak':>6}{'det':>9}{'cut@1.0':>9}{'frac':>7}{'clean':>7}  leak sentence")
    for p in sorted(kept, key=lambda p: (p["leak_detector"] is not None, p["pn"])):
        f = "-" if p["leak_idx"] is None else f"{p['leak_idx']/p['n_sentences']:.2f}"
        print(f"{p['pn']:<6}{p['n_sentences']:>6}{str(p['leak_idx']):>6}{str(p['leak_detector']):>9}"
              f"{p['last_safe_sentence']:>9}{f:>7}{p['clean_cuts'][-1]['n_prefix_sentences']:>7}  "
              f"{(p['leak_sentence'] or '(none found - whole trace handed over)')[:78]}")

    print(f"\n{'='*100}\nFULL TEXT of {args.show} problems (NO-LEAK ones first). Read the last kept sentence "
          f"and the first cut sentence: the first must not state the hinted answer.\n{'='*100}")
    for p in sorted(kept, key=lambda p: (p["leak_detector"] is not None, p["pn"]))[:args.show]:
        s = split_solution_into_chunks(p["reasoning_text"])
        k = p["last_safe_sentence"]
        print(f"\n--- pn {p['pn']} | hinted (WRONG) answer ({p['cue_answer']}) {p['options'][p['cue_answer']][:70]}"
              f"\n    ground truth ({p['gt_answer']}) {p['options'][p['gt_answer']][:70]}"
              f"\n    {len(s)} sentences, keeping {k}, leak at {p['leak_idx']} via {p['leak_detector']}")
        for i in range(max(0, k - 3), min(len(s), k + 3)):
            print(f"    {'KEEP' if i < k else 'CUT '} #{i:<3} {s[i][:200]}")
    if not args.ack:
        print(f"\nCHECKPOINT: {len(kept)} problems selected. Rerun with --ack to accept.")
        sys.exit(2)
    print(f"\nacknowledged: {len(kept)} problems written to {args.out}")


if __name__ == "__main__":
    main()
