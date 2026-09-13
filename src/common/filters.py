"""Semantic filter for the replaced pile: a reader's replacement
sentence counts as a genuine replacement iff its cosine similarity to A's
sentence is below the threshold. Same model, same arithmetic and the same
strict inequality as thought-anchors analyze_rollouts.py: SentenceTransformer
encode with default settings (:1554-1585), cosine = dot / (norm * norm)
(:702-705), dissimilar iff `similarity < threshold` (:708), threshold 0.8
(:34). Runs on CPU so it never competes with vLLM for a card.
"""

import numpy as np


class SemanticFilter:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", threshold: float = 0.8):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device, local_files_only=True)   # cached under HF_HOME; the hub rate-limited us (429) on 2026-09-09
        self.threshold = threshold
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.model.encode([text], show_progress_bar=False)[0]
        return self._cache[text]

    def similarity(self, a: str, b: str) -> float:
        ea, eb = self._embed(a), self._embed(b)
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))

    def is_dissimilar(self, similarity: float) -> bool:
        return similarity < self.threshold
