"""Loader for the released math-rollouts dataset (author A = R1-Distill-Qwen-14B).

Layout, as verified on disk (see doc/setup.md):

    data/math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/
      {correct,incorrect}_base_solution/problem_<id>/
        problem.json gt_answer, gt_solution, level, type, problem, nickname
        base_solution.json prompt, full_cot, solution, answer, is_correct
        chunks.json chunks[], solution_text, source_text
        chunks_labeled.json list, one per chunk (scores, function_tags, ...)
        chunk_<i>/solutions.json 100 stored rollouts resampled at sentence i
"""

import json
from pathlib import Path

from .io import REPO_ROOT

ROLLOUTS_ROOT = (
    REPO_ROOT
    / "data/math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95"
)
SPLITS = ("correct_base_solution", "incorrect_base_solution")


def assert_dataset_present() -> None:
    """Fail early, and with the fix, when the released rollouts are not on disk.

    Without this the first Phase 1 step writes a header-only anchor table, prints a success
    line, and then dies deep inside a sampling call - the empty artefact is the confusing part.
    """
    missing = [s for s in SPLITS if not (ROLLOUTS_ROOT / s).is_dir()]
    if missing:
        raise SystemExit(
            f"{ROLLOUTS_ROOT} has no {', '.join(missing)}. Phase 1 reads the released Thought "
            f"Anchors rollouts; download them first (see the README's Setup):\n"
            f"  .venv/bin/hf download uzaymacar/math-rollouts --repo-type dataset \\\n"
            f"    --local-dir data/math-rollouts --include \"deepseek-r1-distill-qwen-14b/*\"")


def problem_dirs(split: str) -> list[Path]:
    return sorted(
        (ROLLOUTS_ROOT / split).glob("problem_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )


def _load(d: Path, name: str):
    with open(d / name) as f:
        return json.load(f)


class Trace:
    """One problem's released trace: text, per-sentence scores, stored rollouts."""

    def __init__(self, d: Path, split: str):
        self.dir = d
        self.split = split
        self.problem_id = d.name
        self.problem = _load(d, "problem.json")
        self.base = _load(d, "base_solution.json")
        self.chunks_labeled = _load(d, "chunks_labeled.json")
        self._chunks_json = _load(d, "chunks.json")

    @property
    def gt_answer(self) -> str:
        return self.problem["gt_answer"]

    @property
    def question(self) -> str:
        return self.problem["problem"]

    @property
    def prompt(self) -> str:
        """A's exact prompt, ending '... Solution: \\n<think>\\n'. Reader prompts
        are built separately per model (src/common/prompts.py) - this one is A's."""
        return self.base["prompt"]

    @property
    def chunks(self) -> list[str]:
        return self._chunks_json["chunks"]

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def base_is_correct(self) -> bool:
        return _as_bool(self.base["is_correct"])

    def rollouts(self, chunk_idx: int) -> list[dict]:
        """The stored rollouts resampled at sentence chunk_idx (100 per sentence).

        'is_correct' is a bool in every stored rollout (checked on 647,400 by the
        2026-09-09 code review; an earlier note here claiming it was a string was
        wrong). _as_bool accepts either form.
        """
        p = self.dir / f"chunk_{chunk_idx}" / "solutions.json"
        if not p.exists():
            return []
        with open(p) as f:
            return json.load(f)

    def prefix_matches_dataset(self, chunk_idx: int) -> bool:
        """False at the ~3.9% of positions where the stored rollouts were generated
        from a CORRUPTED prefix. generate_rollouts.py:499 builds the prefix as
        full_prefix.replace(chunk, "") - no count - so when sentence i's text
        also occurs earlier in the trace, every copy was deleted from A's earlier
        reasoning before resampling. The stored accuracy/importance at such a
        position is not about chunks[:i]. Excluded from anchors and partners
        (configs/phase1.json, decided 2026-09-08); not recomputed, to
        stay comparable with the paper. Verified 254/6474 by a check."""
        from .prompts import join_prefix

        chunk = self.chunks[chunk_idx]
        generator_prefix = join_prefix(self.chunks[: chunk_idx + 1]).replace(chunk, "").strip()
        return generator_prefix == join_prefix(self.chunks[:chunk_idx])

    def convergence_index(self, k: int = 4, high: float = 0.99, low: float = 0.01) -> int | None:
        """thought-anchors' own convergence rule, plots.py:196-213: the first
        chunk i such that chunks i..i+k-1 ALL have released accuracy > `high`
        (locked on right) or ALL < `low` (locked on wrong). Testable set is
        0..i-1; plots.py:219-221 skips `chunk_idx >= converged_from_index`, and
        `None` (never converges) skips nothing - every sentence is testable.

        k consecutive chunks, not one: each chunk's accuracy is estimated from
        ~100 rollouts, so a single crossing is one draw, not a property of the
        trace. Under the old k=1 modal-agreement>=0.98 rule, agreement fell back
        below threshold within 4 chunks in 28 of 37 'converged' traces and 24 of
        120 top-importance sentences were discarded. decided 2026-09-08;
        Design changelog 2026-09-08. Reads only the released labels - no rollout IO."""
        acc = [c.get("accuracy") for c in self.chunks_labeled]
        for i in range(len(acc) - k - 1):
            window = acc[i:i + k]
            if any(a is None for a in window):
                continue
            if all(a > high for a in window) or all(a < low for a in window):
                return i
        return None

    def is_measured(self, chunk_idx: int) -> bool:
        """False where the released importance is 0.0 because it was never
        measured rather than measured as zero: analyze_rollouts.py:735-736
        returns 0.0 when there are no dissimilar rollouts or no next chunk.
        29.2% of chunks are such zeros; under any magnitude transform they
        would be the most attractive partners. (An earlier citation of
        plots.py:266-279 for this filter was wrong: that counter is
        initialised at :133 and printed at :278 but never incremented.)"""
        if chunk_idx >= self.n_chunks - 1:          # no next chunk
            return False
        return bool(self.chunks_labeled[chunk_idx].get("different_trajectories_fraction"))

    def overdeterminedness(self, chunk_idx: int) -> float:
        """thought-anchors' quantity as its plotting code defines it (plots.py:1029):
        1 - different_trajectories_fraction, the share of the 100 resamples at
        this sentence that reproduced A's sentence (cosine >= 0.8). NOT the
        released `overdeterminedness` field in chunks_labeled.json, which is
        the exact-duplicate ratio (analyze_rollouts.py:721-733) and differs
        from 1 - dtf on 6402/6474 chunks."""
        return 1.0 - self.chunks_labeled[chunk_idx].get("different_trajectories_fraction", 0.0)

    def not_overdetermined(self, chunk_idx: int, max_overdeterminedness: float) -> bool:
        """thought-anchors' own filter in its comparison plots, plots.py:1029-1030
        (also :1685-1686, :1828-1829, :1998-1999): overdeterminedness <= 0.95,
        i.e. at least 5 of the 100 resamples wrote a different sentence. A
        released importance at overdeterminedness 0.99 is one rollout's answer
        minus the next chunk's accuracy. decided 2026-09-08 (the anchor table)."""
        return self.overdeterminedness(chunk_idx) <= max_overdeterminedness

    def importance_a(self, chunk_idx: int) -> float | None:
        """A's released counterfactual importance = P(correct|replaced) −
        P(correct|kept) (analyze_rollouts.py:773, `dissimilar_accuracy -
        next_accuracy`), SIGNED: `abs()` is applied only under --absolute,
        which defaults False (:36). Note the sign order is replaced−kept, the
        opposite of the design's kept−replaced prose.

        Returned signed, the paper's own value. Anchor selection applies its
        own abs() (the transform of its t-test and scatter script,
        sentence_scatter_and_ttests.py:16,74 - decided 2026-09-08 the anchor table)
        at 02_build_anchor_table.py:66. None where not measured."""
        if not self.is_measured(chunk_idx):
            return None
        v = self.chunks_labeled[chunk_idx].get("counterfactual_importance_accuracy")
        if v is None:
            return None
        return v

    def testable(self, chunk_idx: int, k: int = 4, max_overdeterminedness: float = 0.95) -> bool:
        """Every eligibility rule for a sentence position, in one place:
        before convergence, measured, not overdetermined, and its stored
        rollouts AND those of chunk i+1 generated from the true prefix (both
        metrics read chunks i and i+1, so a corrupt i+1 poisons a clean i's
        score)."""
        c = self.convergence_index(k)
        if c is not None and chunk_idx >= c:
            return False
        return (self.is_measured(chunk_idx)
                and self.not_overdetermined(chunk_idx, max_overdeterminedness)
                and self.prefix_matches_dataset(chunk_idx)
                and self.prefix_matches_dataset(chunk_idx + 1))


def _as_bool(v) -> bool:
    """Accepts a bool or the string 'True'/'False'. The release stores real bools
    (647,400/647,400 stored rollout rows checked); the string branch is defensive."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def load_traces(splits=SPLITS) -> list[Trace]:
    return [Trace(d, s) for s in splits for d in problem_dirs(s)]
