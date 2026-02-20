"""Toy MNIST classifier using MTNCLNetwork.

This is intentionally small so MTNCL's simulated-annealing training can finish.
It trains on a tiny balanced subset of MNIST with optional class filtering.

Examples:
  # 2-class toy classifier (0 vs 1)
  python examples/toy_models/mnist_mtncl.py --classes 0,1 --train-per-class 30 --test-per-class 10 --iterations 500

  # 4-class toy classifier (0,1,2,3)
  python examples/toy_models/mnist_mtncl.py --classes 0,1,2,3 --train-per-class 20 --test-per-class 10 --iterations 700
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.datasets import fetch_openml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT, "src"))

from mtncl_nn import MTNCLNetwork


def parse_classes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def downsample_28_to_7(img_flat: np.ndarray) -> np.ndarray:
    # img_flat: shape (784,) in [0..255]
    img = img_flat.reshape(28, 28)
    # 4x4 average pooling -> 7x7
    pooled = img.reshape(7, 4, 7, 4).mean(axis=(1, 3))
    # binarize for gate-friendly inputs
    return (pooled > 64).astype(float).reshape(-1)


def one_hot(index: int, n: int) -> List[float]:
    v = [0.0] * n
    v[index] = 1.0
    return v


def build_subset(
    X: np.ndarray,
    y: np.ndarray,
    classes: List[int],
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    rng = np.random.default_rng(seed)

    X_train, y_train, X_test, y_test = [], [], [], []

    for class_idx, digit in enumerate(classes):
        idx = np.where(y == str(digit))[0]
        rng.shuffle(idx)

        need = train_per_class + test_per_class
        chosen = idx[:need]
        train_idx = chosen[:train_per_class]
        test_idx = chosen[train_per_class:need]

        for i in train_idx:
            X_train.append(downsample_28_to_7(X[i]).tolist())
            y_train.append(one_hot(class_idx, len(classes)))

        for i in test_idx:
            X_test.append(downsample_28_to_7(X[i]).tolist())
            y_test.append(one_hot(class_idx, len(classes)))

    return X_train, y_train, X_test, y_test


def accuracy(model: MTNCLNetwork, X: List[List[float]], y: List[List[float]]) -> float:
    correct = 0
    for x_i, y_i in zip(X, y):
        pred = model.forward(x_i)
        if pred == y_i:
            correct += 1
    return correct / len(X) if X else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classes", type=str, default="0,1")
    p.add_argument("--train-per-class", type=int, default=30)
    p.add_argument("--test-per-class", type=int, default=10)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--hidden-layers", type=str, default="24,12")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-prefix", type=str, default="examples/outputs/toy_models/mnist")
    args = p.parse_args()

    classes = parse_classes(args.classes)
    hidden_layers = [int(x) for x in args.hidden_layers.split(",") if x.strip()]

    print("Loading MNIST from OpenML (cached after first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data
    y = mnist.target

    X_train, y_train, X_test, y_test = build_subset(
        X=X,
        y=y,
        classes=classes,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )

    print(
        f"Training toy MNIST classifier on digits={classes} | "
        f"train={len(X_train)} test={len(X_test)} "
        f"inputs={len(X_train[0])} outputs={len(classes)}"
    )

    model = MTNCLNetwork(
        num_inputs=len(X_train[0]),
        num_outputs=len(classes),
        hidden_layers=hidden_layers,
        verbose=args.verbose,
    )
    model.train(X_train, y_train, iterations=args.iterations)

    train_acc = accuracy(model, X_train, y_train)
    test_acc = accuracy(model, X_test, y_test)

    print(f"Train accuracy: {train_acc * 100:.1f}%")
    print(f"Test accuracy:  {test_acc * 100:.1f}%")

    out_prefix = Path(args.save_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = f"{out_prefix}.json"
    netlist_path = f"{out_prefix}.v"
    model.save(model_path)
    Path(netlist_path).write_text(model.to_verilog(module_name="mnist_mtncl"), encoding="utf-8")
    print(f"Saved model:   {model_path}")
    print(f"Saved netlist: {netlist_path}")


if __name__ == "__main__":
    main()
