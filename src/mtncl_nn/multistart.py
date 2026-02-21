"""Multi-start training helpers for MTNCL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .core.network import MTNCLNetwork


@dataclass
class MultiStartResult:
    best_model: MTNCLNetwork
    best_train_acc: float
    runs: List[dict]


def _acc(model: MTNCLNetwork, X: List[List[float]], y: List[List[float]]) -> float:
    correct = 0
    for x_i, y_i in zip(X, y):
        if model.forward(x_i) == y_i:
            correct += 1
    return correct / len(X) if X else 0.0


def train_multistart(
    num_inputs: int,
    num_outputs: int,
    hidden_layers: List[int],
    X: List[List[float]],
    y: List[List[float]],
    iterations: int,
    starts: int = 5,
) -> MultiStartResult:
    runs = []
    best_model = None
    best_acc = -1.0

    for i in range(starts):
        model = MTNCLNetwork(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            hidden_layers=hidden_layers,
            verbose=False,
        )
        model.train(X, y, iterations=iterations)
        acc = _acc(model, X, y)
        runs.append({"start": i, "train_acc": acc})
        if acc > best_acc:
            best_acc = acc
            best_model = model

    return MultiStartResult(best_model=best_model, best_train_acc=best_acc, runs=runs)
