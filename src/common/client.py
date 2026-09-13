"""vLLM OpenAI-compatible client for one served model.

- `assert_serving()` GET /v1/models must list this model key (
  a client never silently talks to a different model on the same port).
- `complete()` POST /v1/completions, n samples of one fully-built prompt
  string. Always skip_special_tokens=false: gpt-oss's reasoning markers are
  special tokens and the closed/reopened detector needs them in the raw text.
  The terminating stop/EOS token (<|im_end|>, <|return|>) is never in `text`
  (vLLM drops it) but is in the token count.
- `tokenize_text()` / `tokenize_chat()` POST /tokenize, so the probe can
  assert that the server's own chat-template render equals our prompt string
  up to the appended opener (template correctness, checked every run).
Retries with backoff on connection errors and 5xx only. 4xx and read timeouts
raise at once with the server's message; nothing else is hidden.
"""

import time

import requests

from .io import REPO_ROOT, load_config


class VLLMClient:
    def __init__(self, model_key: str, timeout: float = 900.0):
        cfg = load_config(REPO_ROOT / "configs/models.json")
        self.model_key = model_key
        self.base = f"http://{cfg['host']}:{cfg['models'][model_key]['port']}"
        self.timeout = timeout  # 8 x n=60 x 3000 tok on gpt-oss-20b took 50 s

    def _post(self, path: str, payload: dict, retries: int = 5) -> dict:
        for attempt in range(retries):
            try:
                r = requests.post(self.base + path, json=payload, timeout=self.timeout)
            except requests.ConnectionError as e:
                err = f"connection error: {e}"
            except requests.Timeout:
                raise RuntimeError(f"{self.model_key} {path}: no reply in {self.timeout}s; "
                                   "the whole request is lost (vLLM aborts it), not retried") from None
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code < 500:
                    raise RuntimeError(f"{self.model_key} {path} HTTP {r.status_code}: {r.text[:500]}")
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            if attempt == retries - 1:
                raise RuntimeError(f"{self.model_key} {path} failed after {retries} tries: {err}")
            time.sleep(2 ** attempt)

    def assert_serving(self) -> None:
        try:
            data = requests.get(self.base + "/v1/models", timeout=10).json()["data"]
        except requests.ConnectionError:
            raise RuntimeError(f"nothing answering at {self.base}; start it with "
                               f"./serve/serve_model.sh {self.model_key}") from None
        except (ValueError, KeyError):
            raise RuntimeError(f"{self.base}/v1/models did not answer like vLLM") from None
        ids = [m["id"] for m in data]
        if self.model_key not in ids:
            raise RuntimeError(f"{self.base} serves {ids}, not {self.model_key}")

    def complete(self, prompt: str, n: int, max_tokens: int, temperature: float,
                 top_p: float, seed: int | None = None, **extra) -> list[dict]:
        """n raw continuations of one prompt: [{text, finish_reason, tokens}].

        `seed` is honoured by vLLM but does NOT make samples reproducible: the
        logits depend on batch shape and prefix-cache state, so the same seed
        diverges under concurrent load (verified 2026-09-08). A rollout exists
        only as its written row; never regenerate one from its seed. vLLM
        gives sample i the seed `seed + i`, so seeds for different calls must
        be >= n apart - draw them from io.rng(...).getrandbits(31) and store
        them in the row."""
        payload = {"model": self.model_key, "prompt": prompt, "n": n,
                   "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                   "skip_special_tokens": False, "echo": False, "return_token_ids": True,
                   **extra}  # top_k, min_p, presence_penalty, repetition_penalty: vLLM 0.28 CompletionRequest
        if seed is not None:
            payload["seed"] = seed
        out = self._post("/v1/completions", payload)
        # usage is per request; per-choice length = returned token ids (incl. EOS).
        return [{"text": c["text"], "finish_reason": c["finish_reason"],
                 "tokens": len(c["token_ids"])} for c in out["choices"]]

    def tokenize_text(self, text: str) -> list[int]:
        """Same tokenisation /v1/completions applies to `prompt`. No tokenizer
        adds a BOS on this stack (transformers 5.16.1, checked 2026-09-09); A's
        BOS is the literal A_BOS at the start of its prompt (prompts.py)."""
        return self._post("/tokenize", {"model": self.model_key, "prompt": text,
                                        "add_special_tokens": True})["tokens"]

    def tokenize_chat(self, messages: list[dict], enable_thinking: bool = True) -> list[int]:
        """Token ids of the server's own chat-template render of `messages`
        with the assistant turn opened - the ground truth for our prompt prefix."""
        return self._post("/tokenize", {"model": self.model_key, "messages": messages,
                                        "add_generation_prompt": True,
                                        "chat_template_kwargs": {"enable_thinking": enable_thinking}})["tokens"]
