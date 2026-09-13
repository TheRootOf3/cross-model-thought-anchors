# Cross-model thought anchors

Does the sentence-level importance that [Thought Anchors](https://arxiv.org/abs/2506.19143) measures
belong to the *text* of a chain of thought, or to the *model that wrote it*? This repository is the
code behind a short exploratory study of that question, run on DeepSeek-R1-Distill-Qwen-14B's
released MATH traces (Phase 1) and on [Thought Branches](https://arxiv.org/abs/2510.27484)' hinted
multiple-choice traces (Phase 2).

**Note.** This 20-hour project was conducted towards Neel Nanda’s Winter 2027 MATS stream. The number of evaluated models and CoT traces is limited; the findings, while true here, may not apply universally across all reasoning models. 

- **Report** [blog post](https://www.lesswrong.com/posts/EmcEPCFSKKMtCBBkw/thought-anchors-don-t-transfer-between-models).
- **Data:** a separate bundle — [here](https://drive.google.com/drive/folders/1aQog9xv2dYPry2iSRUwFGGIVy2Rlr7ny?usp=drive_link) — with the raw rollouts (258,220 of them,
  688 MB gzipped), the aggregates computed from them, and every figure with the values it plots.
  It is not in this repository; two of its files exceed GitHub's per-file limit.
- **Licence:** MIT, see `LICENSE`. Code copied from upstream repositories and the bundled font carry
  their own notices — see `THIRD_PARTY_NOTICES.md`.

Terminology used throughout: the **writer** (A) is the model that produced a chain of thought —
always R1-Distill-Qwen-14B; a **reader** (B) is a model handed a prefix of A's chain of thought and
asked to continue it — gpt-oss-20b and Qwen3-1.7B.

---

## Setup

```bash
uv sync                              # Python 3.12, exact versions from uv.lock
export HF_HOME=~/.cache/huggingface  # where weights and tokenizers are read from
```

Nothing is fetched from the Hub at run time — `serve/serve_model.sh` exports `HF_HUB_OFFLINE=1` and
the semantic filter loads its encoder with `local_files_only=True` — so pre-fetch everything you will
use, or those steps fail with an error that blames your network rather than this flag:

```bash
.venv/bin/hf download openai/gpt-oss-20b
.venv/bin/hf download Qwen/Qwen3-1.7B
.venv/bin/hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
.venv/bin/hf download sentence-transformers/all-MiniLM-L6-v2     # the semantic filter
```

Those four cover sections 2–5. Reproducing the reader screen in section 1 needs the other six candidates
too — `allenai/Olmo-3-7B-Think`, `Qwen/Qwen3-4B-Thinking-2507` and the four `Qwen/Qwen3.5-*` sizes named in
`configs/models.json`.

Phase 1 reads the released Thought Anchors rollouts (~55 GB; A's own data, not ours):

```bash
.venv/bin/hf download uzaymacar/math-rollouts --repo-type dataset \
  --local-dir data/math-rollouts --include "deepseek-r1-distill-qwen-14b/*"
```

Phase 2 reads Thought Branches' released problem file straight out of their repository. Pin the
commit this study used — `configs/phase2.json` names one file inside it by path:

```bash
git clone https://github.com/interp-reasoning/thought-branches external/thought-branches
git -C external/thought-branches checkout 9e2bba4
```

Some Phase 1 code is copied from `interp-reasoning/thought-anchors` (commit `b53ed8c`), with the
upstream file and line range in a header comment; clone it into `external/thought-anchors` if you
want to diff against it.

Everything runs from the repository root as `.venv/bin/python scripts/NN_name.py …`. Models are
served by vLLM and never loaded inside a script:

```bash
./serve/serve_model.sh <key> --gpu <ids>            # keys and sampling: configs/models.json
./serve/stop_model.sh  <key> --yes                  # only when the jobs using it have finished
```

Hardware used throughout: one machine with 4 × RTX PRO 6000 (97.9 GB), vLLM 0.28.

### Re-analysing without a GPU

The data bundle carries the 258,220 rollouts behind the study's results, over the 19 runs they use; the
superseded attempts and the setup probes are not in it. To redo the analysis and the figures without
regenerating anything, unpack it and put its files where the scripts look:

| from the bundle                                            | to                                  |
|------------------------------------------------------------|-------------------------------------|
| `rollouts/<run_id>/rows.jsonl.gz` (`gunzip`)               | `runs/<run_id>/rows.jsonl`          |
| `rollouts/<run_id>/{config.json,profile.csv,summary.json}` | `runs/<run_id>/`                    |
| `phase1/anchor_table.csv`                                  | `data/anchor_table.csv`             |
| `phase1/selected_traces.json`                              | `data/phase1_traces.json`           |
| `phase1/screen_summary.csv`                                | `data/screen_summary.csv`           |
| `phase2/problems_permissive_45.json`                       | `data/phase2_problems.json`         |
| `phase2/problems_statement_free_41.json`                   | `data/phase2_problems_nostate.json` |
| `phase2/problems_options_never_named_16.json`              | `data/phase2_problems_strict.json`  |
| `phase2/prefix_labels_permissive.json`                     | `data/phase2_prefix_labels.json`    |
| `phase2/prefix_labels_statement_free.json`                 | `data/phase2_nostate_labels.json`   |

One caveat: **A's released rollouts are still needed** — the bundle holds our rollouts, not A's — so run
the `hf download uzaymacar/math-rollouts` step above first. `fig3d`, `fig7b` and `fig2c` read them
directly (they draw A's own accuracy alongside the reader's).

With that in place, the analysis and figure steps below reproduce the data bundle's figures byte for byte
(verified: all ten, PNG and `_data.csv`, from the bundle plus A's released rollouts). Run them by name — a bare
`90_make_figures.py` also tries the setup probes, which need rollouts no bundle carries.

### Figure → command → file

Figures are named by their file stem; the `figNN` numbers are working names, not a reading order.

| figure   | what it shows                                                | command                                   | file                                             |
|----------|--------------------------------------------------------------|-------------------------------------------|--------------------------------------------------|
| `fig3`   | reader screen, all eight candidates                          | `scripts/90_make_figures.py --only fig3`  | `plots/fig3_screen.png`                          |
| `fig3e`  | reader screen, the two readers used                          | `scripts/90_make_figures.py --only fig3e` | `plots/fig3e_screen_selected.png`                |
| `fig2c`  | all 120 anchor/partner pairs, by trace                       | `scripts/90_make_figures.py --only fig2c` | `plots/fig2c_anchor_pairs_by_trace.png`          |
| `fig3d`  | two selected traces in detail                                | `scripts/90_make_figures.py --only fig3d` | `plots/fig3d_two_traces.png`                     |
| `fig2d`  | six random anchor/partner pairs with their d                 | `scripts/90_make_figures.py --only fig2d` | `plots/fig2d_pair_examples.png`                  |
| `fig5`   | Experiment 1: d for every pair, per reader                   | `scripts/08_analyze_anchors.py`           | `plots/fig5_p1_paired.png`                       |
| `fig7b`  | Experiment 2: two dense profiles                             | `scripts/90_make_figures.py --only fig7b` | `plots/fig7b_dense_pair.png`                     |
| `fig12a` | Experiment 2: A's anchors in the reader's importance profile | `scripts/98_fig_anchor_rank.py`           | `plots/fig12a_anchor_strip_gpt-oss-20b.png`      |
| `fig12b` | Experiment 2: every trace's percentile curve                 | `scripts/98_fig_anchor_rank.py`           | `plots/fig12b_percentile_curves_gpt-oss-20b.png` |
| `fig98a` | Experiment 3: effect by cut depth, three arms                | `scripts/99_fig_phase2_ladder.py`         | `plots/fig98a_phase2_ladder_only.png`            |

Every figure writes a `_data.csv` beside it holding exactly the values plotted.

---

## 1. Reader screen — `fig3`, `fig3e`

Which candidate readers are neither saturated nor hopeless on the 20 MATH problems? A reader is kept
if ≥ 10 of the 20 problems land inside a 15–85 % accuracy band — outside it a sentence-level
intervention has almost no room to move the answer. Two of eight candidates qualify.

```bash
./serve/serve_model.sh gpt-oss-20b --gpu 1 --mem-util 0.45 --allow-shared-gpu
mkdir -p runs/screen_gpt-oss-20b
setsid nohup .venv/bin/python scripts/03_screen_readers.py --reader gpt-oss-20b \
  --workers 4 --timeout 14400 > runs/screen_gpt-oss-20b/log.txt 2>&1 &
# …repeat per candidate: qwen3-1.7b, qwen3.5-9b, qwen3.5-4b, qwen3.5-2b, qwen3.5-0.8b,
#   olmo3-7b-think, qwen3-4b-thinking-2507
.venv/bin/python scripts/09_sanity_checks.py --recompute --run_id screen_gpt-oss-20b
./serve/stop_model.sh gpt-oss-20b --yes                     # once the screens have finished
.venv/bin/python scripts/90_make_figures.py --only fig3     # all eight candidates
.venv/bin/python scripts/90_make_figures.py --only fig3e    # the two readers used
```

*Writes* `runs/screen_<key>/rows.jsonl` (20 problems × 40 rollouts), `data/screen_summary.csv`.
*Check* — must print `gpt-oss-20b 11` and `qwen3-1.7b 11`:

```bash
awk -F, 'NR>1 && $11=="True" {c[$1]++} END {for (r in c) print r, c[r]}' data/screen_summary.csv | sort
```

*Cost* ≈ 5,800 rollouts.

## 2. Anchor table and anchor/partner selection — `fig2c`, `fig3d`, `fig2d`

No GPU. Reads A's released per-sentence accuracy and counterfactual importance and picks, per trace,
the top-3 testable sentences in the direction matching the trace outcome (negative — A needed the
sentence — on traces A got right; positive on traces it got wrong), each paired with a
near-zero-importance sentence within ±4. `--ack` is the second invocation: the script prints the
selection and stops, so a human sees it before it is written.

```bash
.venv/bin/python scripts/02_build_anchor_table.py --config configs/phase1.json --run_id h2_anchor_table
.venv/bin/python scripts/02_build_anchor_table.py --config configs/phase1.json --run_id h2_anchor_table --ack
.venv/bin/python scripts/04_select_traces.py --config configs/phase1.json \
  --run_id h3_select_traces --readers qwen3-1.7b gpt-oss-20b
.venv/bin/python scripts/90_make_figures.py --only fig2c    # all 120 pairs
.venv/bin/python scripts/90_make_figures.py --only fig3d    # two-trace case study
.venv/bin/python scripts/90_make_figures.py --only fig2d    # six random pairs (needs section 3's outputs)
```

*Writes* `data/anchor_table.csv` (6,474 sentences, 3,780 testable, 120 anchors + 120 partners) and
`data/phase1_traces.json` (10 traces per reader, 5 A-correct and 5 A-wrong, disjoint problems).
*Check*: `plots/fig2c_anchor_pairs_by_trace_data.csv` has 120 pair rows. *Cost*: seconds, CPU only.

## 3. Experiment 1 — paired anchor/partner test — `fig5`

Per sentence, `importance_B = P(correct | resampled to something dissimilar) − P(correct | kept)`;
per pair, `d = sign_A(anchor) × [importance_B(anchor) − importance_B(partner)]`, so d > 0 means the
reader moved the way A's data predicts, more at the anchor than at its control.

```bash
./serve/serve_model.sh gpt-oss-20b --gpu 0,2 -- --tensor-parallel-size 2
./serve/serve_model.sh qwen3-1.7b  --gpu 3
mkdir -p runs/p1_gpt-oss-20b runs/p1_qwen3-1.7b
setsid nohup .venv/bin/python scripts/07_run_anchor_transfer.py --reader gpt-oss-20b \
  --workers 10 --timeout 7200 > runs/p1_gpt-oss-20b/log.txt 2>&1 &
setsid nohup .venv/bin/python scripts/07_run_anchor_transfer.py --reader qwen3-1.7b \
  --workers 10 --timeout 7200 > runs/p1_qwen3-1.7b/log.txt 2>&1 &
# when both print "done:" in their log.txt:
./serve/stop_model.sh gpt-oss-20b --yes ; ./serve/stop_model.sh qwen3-1.7b --yes
.venv/bin/python scripts/08_analyze_anchors.py     # data/p1_*, doc/p1_results.md, fig5
.venv/bin/python scripts/09_sanity_checks.py --all
```

*Writes* `runs/p1_<reader>/rows.jsonl` (6,520 and 6,800 rollouts), `data/p1_{importance,pairs}.csv`,
`data/p1_results.json`, `plots/fig5_p1_paired.png`.
*Check* — must print `gpt-oss-20b 0.0529 30` and `qwen3-1.7b 0.0613 28`:

```bash
.venv/bin/python -c "import json; d=json.load(open('data/p1_results.json'))['readers']; \
[print(k, round(v['primary_answered']['mean_d'],4), v['primary_answered']['n_pairs']) for k,v in d.items()]"
```

*Cost* ≈ 13,300 rollouts, a few hours on two servers.

**Second-partner robustness check.** Qwen's anchors were also re-tested against a *second* admissible
near-zero-importance partner: mean d +0.057 (14 pairs, 5 A-correct traces, permutation p 0.031)
against +0.113 for the first partner on the 13 anchors the two sets share. The check was run with an earlier version of this pipeline whose
extra flags are no longer part of it; its inputs and outputs ship in the data bundle
(`phase1/second_partner_*`), and the code here does not regenerate it.

## 4. Experiment 2 — dense profile and the anchor percentile — `fig7b`, `fig12a`, `fig12b`

Re-measures importance at **every** sentence of the same 10 traces, 100 rollouts each, for
gpt-oss-20b only. Qwen3-1.7B was not profiled: it writes ~20k tokens per rollout, and the attempt was stopped after
measuring 3 sentences in an hour — about 14 h for that one trace.

```bash
./serve/serve_model.sh gpt-oss-20b --gpu 0,2 -- --tensor-parallel-size 2
mkdir -p runs/dense_gpt-oss-20b runs/dense_gpt-oss-20b_wrong
# one job at a time on this server; the bundle's runs were sequential
setsid nohup .venv/bin/python scripts/20_dense_profile.py --reader gpt-oss-20b \
  --split correct_base_solution --n 100 --workers 10 --timeout 7200 \
  > runs/dense_gpt-oss-20b/log.txt 2>&1 &                       # 685 sentences, 6 h 35 min
setsid nohup .venv/bin/python scripts/20_dense_profile.py --reader gpt-oss-20b \
  --split incorrect_base_solution --run_id dense_gpt-oss-20b_wrong --n 100 --workers 10 \
  --timeout 7200 > runs/dense_gpt-oss-20b_wrong/log.txt 2>&1 &  # 774 sentences, 8 h 6 min
# when both logs print "done:" (≈ 14 h 45 min in total):
./serve/stop_model.sh gpt-oss-20b --yes
.venv/bin/python scripts/24_anchor_permutation.py    # within-trace label permutation, B = 20,000
.venv/bin/python scripts/98_fig_anchor_rank.py       # fig12a and fig12b
.venv/bin/python scripts/90_make_figures.py --only fig7b   # two dense profiles
```

*Writes* `runs/dense_gpt-oss-20b{,_wrong}/{rows.jsonl,profile.csv}` (1,459 sentences, 145,900
rollouts, 2.7 GB raw) and `runs/h24_anchor_permutation/{result.json,per_trace.csv}`.
*Check* — must print `0.5309 0.0569 0.5768`:

```bash
.venv/bin/python -c "import json; d=json.load(open('runs/h24_anchor_permutation/result.json')); \
print(round(d['observed_mean_percentile'],4), round(d['observed_se'],4), round(d['p_two_sided'],4))"
```

## 5. Experiment 3 — Phase 2 nudge transplant — `fig98a`

Three cut arms over the same source problems, transplanted into a reader that never saw the hint.
All three come from `11_select_traces_p2.py`, which dedupes the released file (71 records → 58
problems), drops problems whose options are not lexically separable or whose reasoning verbalises the
hint, finds the sentence that gives the hinted answer away, and cuts before it.

```bash
# selection (CPU only). The two narrower arms were produced by exactly these commands.
# pn 877 is dropped by hand from the statement-free arm: a rater found the hinted answer stated at
# sentence 14, while the lexical rule only fires 17 sentences later, so no margin can clean it.
.venv/bin/python scripts/11_select_traces_p2.py --rule commit --margin 4 --min_prefix 3 \
  --exclude_pn 877 --out data/phase2_problems_nostate.json --run_id h11_p2_nostate --ack     # 41
.venv/bin/python scripts/11_select_traces_p2.py --rule any_option --min_prefix 4 \
  --out data/phase2_problems_strict.json --run_id h11_p2_strict --ack                        # 16
# The permissive 45 is this script's default output, data/phase2_problems.json. Take it from the data
# bundle rather than re-deriving it: the cut rule was tightened after that arm was generated, so a
# fresh run moves the cut point on 14 of the 45 problems and no longer reproduces the bundle's numbers.

./serve/serve_model.sh gpt-oss-20b   --gpu 0,2 -- --tensor-parallel-size 2
./serve/serve_model.sh r1-distill-14b --gpu 1
for M in gpt-oss-20b r1-distill-14b; do
  mkdir -p runs/p2f_$M runs/p2g_$M runs/p2c_$M
  nohup .venv/bin/python scripts/12_run_transplant.py --model $M --run_id p2f_$M \
    --workers 10 > runs/p2f_$M/log.txt 2>&1 &
  nohup .venv/bin/python scripts/12_run_transplant.py --model $M --run_id p2g_$M \
    --problems data/phase2_problems_nostate.json --workers 10 > runs/p2g_$M/log.txt 2>&1 &
  nohup .venv/bin/python scripts/12_run_transplant.py --model $M --run_id p2c_$M \
    --problems data/phase2_problems_strict.json --workers 10 > runs/p2c_$M/log.txt 2>&1 &
done
# when the six logs print "done:" (≈ 20 min per arm per model):
./serve/stop_model.sh gpt-oss-20b --yes ; ./serve/stop_model.sh r1-distill-14b --yes
.venv/bin/python scripts/99_fig_phase2_ladder.py     # fig98a
```

*Writes* `runs/p2{f,g,c}_<model>/rows.jsonl` (50 samples per cell, 5 cut fractions, 32,768-token cap)
and `plots/fig98a_phase2_ladder_only.{png,_data.csv}`.
*Check* — the deepest cut of the permissive arm must read `43.958`:

```bash
grep '^answer may be stated,gpt-oss-20b,primary,1.0,' plots/fig98a_phase2_ladder_only_data.csv
```

*Cost* 45,900 rollouts per model across the three arms (20,250 + 18,450 + 7,200), about 47 min of
generation per model on one server.

---

## Other scripts

Sections 1–5 are the reproduction path. The rest are prep, diagnostics and checks, kept because the
study's caveats and robustness checks rest on them.

| script                         | what it does                                                                                                                                                |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `01_prefill_probe.py`          | the setup probe: will a reader *continue* a prefilled reasoning block rather than restart it, and does the token cap bind? Both phases depend on the answer |
| `05_calibrate_on_A.py`         | calibration: run the Phase 1 measurement with A as its own reader and compare against A's released importances                                              |
| `06_smoke_test.py`             | one reader, one trace, 8 rollouts per pile, through the full Phase 1 pile code — the cheap end-to-end check before a grid                                   |
| `09_sanity_checks.py`          | independent recomputes of every headline number straight from the raw JSONL, plus the raw-row displays; appends to `doc/checks.md`                          |
| `13_analyze_transplant.py`     | the Phase 2 readout in table form (the figure script is `99`)                                                                                               |
| `15_export_decisions.py`       | every Phase 2 selection and classification decision as one CSV, for human review                                                                            |
| `22_gate_statistics.py`        | what the gate's permutation actually tests, its realised size, and the alternative tests it should be quoted beside                                         |
| `23_profile_checks.py`         | profile-level checks on the dense run: signal-to-noise, the paired result re-measured on independent rollouts                                               |
| `24_anchor_permutation.py`     | the within-trace label permutation behind the anchor-percentile result (section 4)                                                                          |
| `91_sample_examples.py`        | seeded random raw rollouts and pairs, verbatim, for reading rather than summarising                                                                         |
| `94_check_gate_calibration.py` | simulates the gate under a complete null to measure its false-positive rate                                                                                 |
| `95_bos_ab.py`                 | diagnostic: does prepending A's BOS token change the piles? (needs a served model)                                                                          |

---

## Conventions

- `configs/phase1.json` and `configs/phase2.json` hold the thresholds, sample counts and the
  pre-registered gate, each with its reason in a `_note` field; `configs/models.json` holds each
  model's recommended sampling settings and vLLM launch flags. Nothing in `configs/` was changed
  after the data behind a number existed.
- Scripts that make a selection print it and stop; re-run with `--ack` to write it.
- `data/`, `runs/`, `plots/` and `doc/` ship empty (only a `.gitkeep`): they are where outputs land, and
  `.gitignore` keeps them out of version control. `doc/` collects the markdown the analysis writes —
  `p1_results.md`, `checks.md`, the frozen prompt strings in `setup.md`.
- Every run directory holds `config.json` (the exact config used), `log.txt`, and `rows.jsonl` —
  one JSON object per rollout, written as it is produced, never rewritten. Re-running a script
  resumes from what is already on disk rather than regenerating it.
- **Prompt dates.** gpt-oss-20b's chat template writes the current date into every prompt
  (`Current date: YYYY-MM-DD`); the Qwen3-1.7B and R1-Distill prompts carry no date. A run pins the
  date of its first launch and reuses it when resumed, so a run's prompts never change midway; a fresh
  run uses the day it starts. The published gpt-oss runs were rendered on 2026-09-09 (reader screen,
  Experiment 1, the correct-trace dense profile) and 2026-09-10 (the wrong-trace dense profile and all
  three Phase 2 arms), so a rerun's prompts differ from theirs in that one line. Expect the bundle's
  numbers back within sampling noise, not byte-identical rollouts.
- Every random choice goes through the seeded RNG in `src/common/io.py`, so a rerun with the same
  seed makes the same choices.
- Where an experiment went through several attempts, the recipes above give only the final run ids —
  the ones the data bundle holds.

## Credits and citation

This work is built on two papers by the interp-reasoning group, and reuses their measurement code
(copied with the upstream file and line range recorded in each header comment — see
`THIRD_PARTY_NOTICES.md`):

- Bogdan, Macar, Nanda & Conmy, *Thought Anchors: Which LLM Reasoning Steps Matter?*
  (arXiv:2506.19143) — the resampling-importance method, the released MATH rollouts
  (`uzaymacar/math-rollouts`), and the testability rules this code reproduces.
- Macar, Bogdan, Rajamanoharan & Nanda, *Thought Branches: Interpreting LLM Reasoning Requires
  Resampling* (arXiv:2510.27484) — the hinted-CoT transplant design and the released problem file
  Phase 2 reads, itself derived from Chua & Evans' faithfulness results.

If you use this code, please cite those papers as well.
