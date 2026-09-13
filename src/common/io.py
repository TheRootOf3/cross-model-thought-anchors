"""Run plumbing: config load, run dirs, seeded RNG, JSONL append/resume.

Every script in scripts/ goes through here so that (a) nothing is held in
memory only, (b) every random choice is reproducible from the config seed, and
(c) an interrupted run resumes instead of restarting.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_dir(run_id: str, create: bool = True) -> Path:
    d = REPO_ROOT / "runs" / run_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_config(run_id: str, config: dict, extra: dict | None = None) -> Path:
    """Write runs/<run_id>/config.json FIRST, before any generation.
    Never overwritten on resume - the original config is what the run means."""
    p = run_dir(run_id) / "config.json"
    if not p.exists():
        payload = dict(config)
        if extra:
            payload["_run"] = extra
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


def rng(seed: int, *salt: Any) -> random.Random:
    """Seeded RNG. Pass salt strings so different choices in one run get
    different streams while staying reproducible: rng(seed, 'trace_pick')."""
    return random.Random(f"{seed}|" + "|".join(map(str, salt)))


def append_row(path: str | Path, row: dict) -> None:
    """Append one JSON row and flush. One row per rollout, written immediately."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # A killed run can leave a truncated last line; read_rows skips it, but a
    # row appended straight after it would be glued on and lost too.
    # Terminate the stray line first.
    needs_newline = False
    if p.exists() and p.stat().st_size > 0:
        with open(p, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    with open(p, "a") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_rows(path: str | Path) -> Iterator[dict]:
    """Yield rows, skipping a truncated final line (a killed run can leave one)."""
    p = Path(path)
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
