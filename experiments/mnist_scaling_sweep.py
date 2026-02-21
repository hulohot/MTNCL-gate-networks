"""MNIST scaling sweep for MTNCL toy models.

Outputs:
  - experiments/results/mnist_scaling/results.json
  - experiments/results/mnist_scaling/results.csv
  - experiments/results/mnist_scaling/report.md
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.datasets import fetch_openml

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mtncl_nn import MTNCLNetwork


@dataclass
class SweepConfig:
    name: str
    classes: List[int]
    train_per_class: int
    test_per_class: int
    hidden_layers: List[int]
    iterations: int
    seed: int = 42


def downsample_28_to_7(img_flat: np.ndarray) -> np.ndarray:
    img = img_flat.reshape(28, 28)
    pooled = img.reshape(7, 4, 7, 4).mean(axis=(1, 3))
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
    c = 0
    for x_i, y_i in zip(X, y):
        if model.forward(x_i) == y_i:
            c += 1
    return c / len(X) if X else 0.0


def gate_counts(model: MTNCLNetwork) -> tuple[int, int, int]:
    total = sum(len(layer) for layer in model.gates)
    inputs_plus_consts = len(model.gates[0])
    logic = total - inputs_plus_consts
    outputs = len(model.output_gates)
    return total, logic, outputs


def run_xor(cfg: SweepConfig, models_dir: Path) -> dict:
    """Run XOR baseline experiment."""
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[1, 0], [0, 1], [0, 1], [1, 0]]  # XOR -> one-hot
    X_test = X_train[:]
    y_test = y_train[:]
    
    model = MTNCLNetwork(
        num_inputs=2,
        num_outputs=2,
        hidden_layers=cfg.hidden_layers,
        verbose=False,
    )
    model.train(X_train, y_train, iterations=cfg.iterations)
    
    train_acc = accuracy(model, X_train, y_train)
    test_acc = accuracy(model, X_test, y_test)
    total_g, logic_g, out_g = gate_counts(model)
    
    model_path = models_dir / f"{cfg.name}.json"
    netlist_path = models_dir / f"{cfg.name}.v"
    dot_path = models_dir / f"{cfg.name}.dot"
    model.save(str(model_path))
    netlist_path.write_text(model.to_verilog(module_name=f"mnist_{cfg.name}"), encoding="utf-8")
    dot_path.write_text(model.to_dot(graph_name=f"mnist_{cfg.name}"), encoding="utf-8")
    
    return {
        **asdict(cfg),
        "num_classes": 2,
        "train_samples": 4,
        "test_samples": 4,
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "total_gates": total_g,
        "logic_gates": logic_g,
        "output_gates": out_g,
        "netlist_bytes": netlist_path.stat().st_size,
        "dot_bytes": dot_path.stat().st_size,
        "model_path": str(model_path.relative_to(ROOT)),
        "netlist_path": str(netlist_path.relative_to(ROOT)),
        "dot_path": str(dot_path.relative_to(ROOT)),
    }


def run_mnist(cfg: SweepConfig, X: np.ndarray, y: np.ndarray, models_dir: Path) -> dict:
    """Run MNIST experiment."""
    X_train, y_train, X_test, y_test = build_subset(
        X, y, cfg.classes, cfg.train_per_class, cfg.test_per_class, cfg.seed
    )

    model = MTNCLNetwork(
        num_inputs=len(X_train[0]),
        num_outputs=len(cfg.classes),
        hidden_layers=cfg.hidden_layers,
        verbose=False,
    )
    model.train(X_train, y_train, iterations=cfg.iterations)

    train_acc = accuracy(model, X_train, y_train)
    test_acc = accuracy(model, X_test, y_test)
    total_g, logic_g, out_g = gate_counts(model)

    model_path = models_dir / f"{cfg.name}.json"
    netlist_path = models_dir / f"{cfg.name}.v"
    dot_path = models_dir / f"{cfg.name}.dot"
    model.save(str(model_path))
    netlist_path.write_text(model.to_verilog(module_name=f"mnist_{cfg.name}"), encoding="utf-8")
    dot_path.write_text(model.to_dot(graph_name=f"mnist_{cfg.name}"), encoding="utf-8")

    return {
        **asdict(cfg),
        "num_classes": len(cfg.classes),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "total_gates": total_g,
        "logic_gates": logic_g,
        "output_gates": out_g,
        "netlist_bytes": netlist_path.stat().st_size,
        "dot_bytes": dot_path.stat().st_size,
        "model_path": str(model_path.relative_to(ROOT)),
        "netlist_path": str(netlist_path.relative_to(ROOT)),
        "dot_path": str(dot_path.relative_to(ROOT)),
    }


def main() -> None:
    out_dir = ROOT / "experiments" / "results" / "mnist_scaling"
    models_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        # XOR baseline for comparison
        SweepConfig("xor_baseline", [-1], 0, 0, [4, 2], 800),
        # 2-class variants
        SweepConfig("2class_tiny", [0, 1], 20, 10, [16, 8], 400),
        SweepConfig("2class_small", [0, 1], 30, 10, [24, 12], 600),
        # 4-class variants  
        SweepConfig("4class_tiny", [0, 1, 2, 3], 15, 8, [32, 16], 500),
        SweepConfig("4class_small", [0, 1, 2, 3], 25, 10, [40, 20], 700),
    ]

    print("Loading MNIST (OpenML, parser=liac-arff)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data
    y = mnist.target

    rows = []
    for cfg in configs:
        print(f"\n=== Running {cfg.name} ===")
        if cfg.name == "xor_baseline":
            row = run_xor(cfg, models_dir)
        else:
            row = run_mnist(cfg, X, y, models_dir)
        rows.append(row)

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown report
    md = []
    md.append("# MNIST Scaling Sweep (MTNCL)\n")
    md.append("This report measures toy MNIST performance versus exported circuit size for MTNCL models.\n")
    md.append("\n## Setup\n")
    md.append("- Input preprocessing: 28x28 -> 7x7 average pool, then binarize (>64)\n")
    md.append("- Training algorithm: MTNCL simulated annealing\n")
    md.append("- Each run uses a small balanced subset of MNIST\n")
    md.append("\n## Results\n")
    md.append("| config | classes | train/test per class | hidden | iters | train acc | test acc | logic gates | total gates | netlist bytes |\n")
    md.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            f"| {r['name']} | {r['num_classes']} | {r['train_per_class']}/{r['test_per_class']} | {r['hidden_layers']} | {r['iterations']} | "
            f"{r['train_acc']*100:.1f}% | {r['test_acc']*100:.1f}% | {r['logic_gates']} | {r['total_gates']} | {r['netlist_bytes']} |\n"
        )

    md.append("\n## Comparison\n")
    md.append("### Accuracy vs Problem Complexity\n")
    for r in rows:
        md.append(f"- **{r['name']}**: {r['num_classes']} classes, {r['test_acc']*100:.1f}% test accuracy\n")
    
    md.append("\n### Circuit Size vs Problem Complexity\n")
    for r in rows:
        md.append(f"- **{r['name']}**: {r['logic_gates']} logic gates, {r['netlist_bytes']} bytes netlist\n")

    md.append("\n## Standard Reference\n")
    md.append("For comparison, typical small MLPs on full MNIST (10 classes, 60k train / 10k test):\n")
    md.append("- Small MLP (~100k params): 97-98% accuracy\n")
    md.append("- LeNet-style CNN (~50k params): 99%+ accuracy\n")
    md.append("- Digital circuit equivalent: thousands to millions of gates\n")
    md.append("\nThese MTNCL toy models are orders of magnitude smaller and run on tiny subsets.\n")

    md.append("\n## Notes\n")
    md.append("- These are **toy** runs, not full MNIST benchmarks.\n")
    md.append("- Small test sets can inflate variance; compare trends, not absolute SOTA numbers.\n")
    md.append("- Artifacts (model/netlist/dot) are stored in `experiments/results/mnist_scaling/models/`.\n")

    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\nWrote results to: {out_dir}")


if __name__ == "__main__":
    main()
