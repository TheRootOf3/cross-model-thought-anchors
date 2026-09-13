#!/usr/bin/env python
"""the calibration run diagnostic (2026-09-09): does prepending A's BOS token change the piles?

    python scripts/95_bos_ab.py --run_id h4_bos_ab [--anchor 44] [--trace incorrect_base_solution/problem_6596]

Background: a check found that our vLLM server tokenises A's raw
prompt WITHOUT the BOS token (transformers 5.x loads the R1-Distill tokenizer
with add_bos_token=False; /tokenize shows no id 151646), while the stack that
generated the dataset (transformers 4.x era, Novita API) would have prepended
BOS - and the project's recorded decision was "A runs with BOS". This script
regenerates one anchor's kept and replaced piles with the literal BOS string
prepended to the otherwise identical prompt (the tokenizer maps it to id
151646), through the same pile code, so the result can be compared pile by
pile with runs/h4_calibrate_A. Diagnostic only; nothing else reads its rows.

Ran once, 2026-09-09 14:33-14:55 (runs/h4_bos_ab). Since then a_raw_prompt
carries A_BOS itself (decided 2026-09-09), so rerunning this script
would prepend a second BOS and trip the single-BOS assert below - intended.
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.phase1.piles as piles
from src.common.client import VLLMClient
from src.common.data import ROLLOUTS_ROOT, Trace
from src.common.filters import SemanticFilter
from src.common.io import load_config, run_dir, write_run_config
from src.common.sampling import reader_sampling

A, BOS = "r1-distill-14b", "<｜begin▁of▁sentence｜>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="h4_bos_ab")
    ap.add_argument("--trace", default="incorrect_base_solution/problem_6596")
    ap.add_argument("--anchor", type=int, default=44)
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()
    cfg = load_config("configs/phase1.json")
    split, pid = args.trace.split("/")
    trace = Trace(ROLLOUTS_ROOT / split / pid, split)
    samp = reader_sampling(cfg, A)
    client = VLLMClient(A, timeout=args.timeout)
    client.assert_serving()
    original = piles.pile_prompt
    piles.pile_prompt = lambda reader, tr, i, pile: BOS + original(reader, tr, i, pile)   # the only change
    toks = client.tokenize_text(piles.pile_prompt(A, trace, args.anchor, "kept"))
    assert toks[0] == 151646 and toks.count(151646) == 1, f"BOS not where expected: {toks[:3]}"
    write_run_config(args.run_id, cfg, extra={"script": "95_bos_ab", "args": vars(args), "sampling_used": samp,
                                              "bos_token_id": toks[0], "note": "prompts = BOS + the calibration prompts"})
    imp = cfg["importance"]
    filt = SemanticFilter(imp["similarity_model"], imp["similarity_device"], imp["similarity_max_cosine"])
    rows_path = run_dir(args.run_id) / "rows.jsonl"
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {pile: ex.submit(piles.ensure_pile, client, cfg, A, trace, args.anchor, pile, samp, rows_path, args.run_id, "bos_ab", filt, lock)
                for pile in piles.PILES}
        out = {pile: f.result() for pile, f in futs.items()}
    res = piles.importance(out["kept"], out["replaced"])
    base = list(piles.read_rows(Path("runs/h4_calibrate_A/rows.jsonl")))
    ref = piles.importance([r for r in base if r["chunk_idx"] == args.anchor and r["pile"] == "kept"],
                           [r for r in base if r["chunk_idx"] == args.anchor and r["pile"] == "replaced"])
    for name, r in (("with BOS", res), ("without BOS (h4_calibrate_A)", ref)):
        k, d = r["kept"], r["replaced"]
        print(f"{name:<30} kept {k['p_correct']:.3f} (n {k['n_answered']})  replaced-all n {d['n']}  dissimilar {d['n_dissimilar']}  "
              f"P(correct|dissimilar) {d['p_correct']:.3f} (n {d['n_answered']})  importance {r['importance']:+.3f}")
    (run_dir(args.run_id) / "summary.json").write_text(json.dumps({"anchor": args.anchor, "with_bos": res, "without_bos": ref}, indent=2))


if __name__ == "__main__":
    main()
