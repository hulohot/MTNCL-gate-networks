"""Simple ensemble wrapper for MTNCL models."""

from __future__ import annotations

from typing import List

from .core.network import MTNCLNetwork


class MTNCLEnsemble:
    def __init__(self, models: List[MTNCLNetwork]):
        self.models = models

    def predict_one(self, x: List[float]) -> List[float]:
        if not self.models:
            return []
        n_out = len(self.models[0].output_gates)
        votes = [0] * n_out
        for m in self.models:
            pred = m.forward(x)
            idx = pred.index(1.0) if 1.0 in pred else 0
            votes[idx] += 1
        best = max(range(n_out), key=lambda i: votes[i])
        return [1.0 if i == best else 0.0 for i in range(n_out)]

    def accuracy(self, X: List[List[float]], y: List[List[float]]) -> float:
        c = 0
        for x_i, y_i in zip(X, y):
            if self.predict_one(x_i) == y_i:
                c += 1
        return c / len(X) if X else 0.0
