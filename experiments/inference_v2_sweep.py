"""Inference-v2 sweep: richer features, curriculum, multistart, ensembles, repeated seeds.

Resume-safe behavior:
- Reuses existing per-seed model artifacts when present.
- Writes incremental checkpoints after each (stage, feature_mode).

Outputs to experiments/results/inference_v2/.
"""

from __future__ import annotations

import json
import csv
import argparse
from pathlib import Path
from statistics import mean, pstdev
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from sklearn.datasets import fetch_openml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mtncl_nn import build_features, train_multistart, MTNCLEnsemble, MTNCLNetwork


@dataclass
class Stage:
    name: str
    classes: List[int]
    train_per_class: int
    test_per_class: int
    hidden_layers: List[int]
    iterations: int


def one_hot(i: int, n: int) -> List[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


def make_split(X, y, classes, train_per_class, test_per_class, seed, feature_mode):
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = [], [], [], []
    for ci, d in enumerate(classes):
        idx = np.where(y == str(d))[0]
        rng.shuffle(idx)
        chosen = idx[: train_per_class + test_per_class]
        tri = chosen[:train_per_class]
        tei = chosen[train_per_class:]
        for i in tri:
            Xtr.append(build_features(X[i], mode=feature_mode))
            ytr.append(one_hot(ci, len(classes)))
        for i in tei:
            Xte.append(build_features(X[i], mode=feature_mode))
            yte.append(one_hot(ci, len(classes)))
    return Xtr, ytr, Xte, yte


def eval_acc(model, X, y):
    c = 0
    for x_i, y_i in zip(X, y):
        if model.forward(x_i) == y_i:
            c += 1
    return c / len(X) if X else 0.0


def gate_counts(model):
    total = sum(len(layer) for layer in model.gates)
    logic = total - len(model.gates[0])
    return logic, total


def key_for(stage_name: str, mode: str) -> str:
    return f"{stage_name}::{mode}"


def write_outputs(out: Path, rows: List[dict]) -> None:
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    md = ["# Inference v2 Sweep\n\n"]
    md.append("Implemented improvements: richer encoding, curriculum stages, multistart training, ensembles, repeated-seed stats.\n\n")
    if rows:
        md.append("| stage | classes | feature | test mean ± std | ensemble | logic gates mean |\n")
        md.append("|---|---:|---|---:|---:|---:|\n")
        for r in rows:
            md.append(
                f"| {r['stage']} | {r['classes']} | {r['feature_mode']} | {r['test_acc_mean']*100:.1f}% ± {r['test_acc_std']*100:.1f}% | {r['ensemble_acc']*100:.1f}% | {r['logic_gates_mean']} |\n"
            )

    md.append("\n## Takeaways\n")
    md.append("- Hybrid/multibit features should outperform pure binary as class count grows.\n")
    md.append("- Variance across seeds remains high at low sample counts; use mean/std for comparisons.\n")
    md.append("- Ensemble voting often stabilizes small-circuit predictions.\n")
    (out / "report.md").write_text("".join(md), encoding="utf-8")


def train_or_load_seed_model(
    models: Path,
    model_name: str,
    Xtr, ytr,
    num_outputs: int,
    hidden_layers: List[int],
    iterations: int,
    multistarts: int,
) -> MTNCLNetwork:
    model_path = models / f"{model_name}.json"
    if model_path.exists():
        return MTNCLNetwork.load(str(model_path))

    ms = train_multistart(
        num_inputs=len(Xtr[0]),
        num_outputs=num_outputs,
        hidden_layers=hidden_layers,
        X=Xtr,
        y=ytr,
        iterations=iterations,
        starts=multistarts,
    )
    model = ms.best_model
    model.save(str(model_path))
    (models / f"{model_name}.v").write_text(model.to_verilog(module_name=model_name), encoding="utf-8")
    (models / f"{model_name}.dot").write_text(model.to_dot(graph_name=model_name), encoding="utf-8")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default=None, help="Optional stage name filter")
    ap.add_argument("--mode", type=str, default=None, help="Optional feature mode filter")
    args = ap.parse_args()
    out = ROOT / "experiments" / "results" / "inference_v2"
    models = out / "models"
    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST...")
    mn = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X, y = mn.data, mn.target

    stages = [
        Stage("s1_2class_easy", [0, 1], 30, 15, [24, 12], 350),
        Stage("s2_4class", [0, 1, 2, 3], 20, 8, [32, 16], 450),
        Stage("s3_10class_tiny", list(range(10)), 6, 3, [48, 24], 350),
    ]

    feature_modes = ["binary", "multibit2", "hybrid2"]
    seeds = [11, 22]
    multistarts = 2
    ensemble_k = 2

    existing_rows: Dict[str, dict] = {}
    results_path = out / "results.json"
    if results_path.exists():
        try:
            for row in json.loads(results_path.read_text(encoding="utf-8")):
                existing_rows[key_for(row["stage"], row["feature_mode"])] = row
        except Exception:
            pass

    for stage in stages:
        if args.stage and stage.name != args.stage:
            continue
        for mode in feature_modes:
            if args.mode and mode != args.mode:
                continue
            k = key_for(stage.name, mode)
            if k in existing_rows:
                print(f"[resume] skipping completed combo: {k}")
                continue

            print(f"Running {k} ...")
            seed_test_accs = []
            seed_train_accs = []
            seed_logic = []
            saved_models: List[Tuple[MTNCLNetwork, List[List[float]], List[List[float]]]] = []

            for seed in seeds:
                Xtr, ytr, Xte, yte = make_split(
                    X, y, stage.classes, stage.train_per_class, stage.test_per_class, seed, mode
                )

                model_name = f"{stage.name}_{mode}_seed{seed}"
                model = train_or_load_seed_model(
                    models=models,
                    model_name=model_name,
                    Xtr=Xtr,
                    ytr=ytr,
                    num_outputs=len(stage.classes),
                    hidden_layers=stage.hidden_layers,
                    iterations=stage.iterations,
                    multistarts=multistarts,
                )

                tr_acc = eval_acc(model, Xtr, ytr)
                te_acc = eval_acc(model, Xte, yte)
                logic, _ = gate_counts(model)

                seed_train_accs.append(tr_acc)
                seed_test_accs.append(te_acc)
                seed_logic.append(logic)
                saved_models.append((model, Xte, yte))

            ens_models = [m for m, _, _ in saved_models[:ensemble_k]]
            ens = MTNCLEnsemble(ens_models)
            ens_X, ens_y = saved_models[0][1], saved_models[0][2]
            ens_acc = ens.accuracy(ens_X, ens_y)

            row = {
                "stage": stage.name,
                "classes": len(stage.classes),
                "feature_mode": mode,
                "train_per_class": stage.train_per_class,
                "test_per_class": stage.test_per_class,
                "hidden_layers": stage.hidden_layers,
                "iterations": stage.iterations,
                "seeds": seeds,
                "multistarts": multistarts,
                "train_acc_mean": round(mean(seed_train_accs), 4),
                "train_acc_std": round(pstdev(seed_train_accs), 4),
                "test_acc_mean": round(mean(seed_test_accs), 4),
                "test_acc_std": round(pstdev(seed_test_accs), 4),
                "ensemble_acc": round(ens_acc, 4),
                "logic_gates_mean": round(mean(seed_logic), 2),
            }
            existing_rows[k] = row

            ordered_rows = [existing_rows[key_for(s.name, m)]
                            for s in stages for m in feature_modes
                            if key_for(s.name, m) in existing_rows]
            write_outputs(out, ordered_rows)
            print(f"[checkpoint] wrote {len(ordered_rows)} rows")

    ordered_rows = [existing_rows[key_for(s.name, m)]
                    for s in stages for m in feature_modes
                    if key_for(s.name, m) in existing_rows]
    write_outputs(out, ordered_rows)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
