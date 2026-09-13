"""Reader sampling settings (design changelog 2026-09-08): each model
samples with its own published recommended settings from configs/models.json,
under the project-wide cap from configs/phase1.json `reader_sampling`."""

from .io import REPO_ROOT, load_config

# vLLM 0.28 CompletionRequest sampling fields we may pass beyond T / top-p / max_tokens
# (entrypoints/openai/completion/protocol.py); anything else would be accepted and ignored.
EXTRA_KEYS = {"top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty"}


def reader_sampling(cfg: dict, model_key: str) -> dict:
    """{'temperature', 'top_p', 'max_tokens', ...extra vLLM sampling keys} for one model.
    A (r1-distill-14b) keeps T 0.6 / top-p 0.95, which are both its own and the dataset's."""
    models = load_config(REPO_ROOT / "configs/models.json")["models"]
    if not cfg["reader_sampling"].get("per_model", False):
        raise ValueError("configs/phase1.json reader_sampling.per_model is False; the pre-registered design says per-model settings")
    s = {k: v for k, v in models[model_key]["recommended_sampling"].items() if not k.startswith("_")}
    rs = cfg["reader_sampling"]
    s["max_tokens"] = rs["max_tokens"] if "max_tokens" in rs else rs["max_tokens_math"]   # phase2 names it max_tokens (MMLU, not MATH)
    return s


def split_sampling(s: dict) -> tuple[float, float, int, dict]:
    """(temperature, top_p, max_tokens, extra) in the shape VLLMClient.complete takes."""
    extra = {k: v for k, v in s.items() if k not in ("temperature", "top_p", "max_tokens")}
    unknown = set(extra) - EXTRA_KEYS
    assert not unknown, f"recommended_sampling keys {unknown} are not vLLM sampling fields (the server ignores unknown keys silently)"
    return s["temperature"], s["top_p"], s["max_tokens"], extra
