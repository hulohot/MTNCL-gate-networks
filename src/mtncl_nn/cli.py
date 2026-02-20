"""CLI utilities for training and inference with MTNCL networks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List

from .core.network import MTNCLNetwork


def _load_dataset(path: str) -> tuple[List[List[float]], List[List[float]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["X"], data["y"]


def cmd_train(args: argparse.Namespace) -> int:
    X, y = _load_dataset(args.data)
    model = MTNCLNetwork(
        num_inputs=args.num_inputs,
        num_outputs=args.num_outputs,
        hidden_layers=args.hidden_layers,
        verbose=args.verbose,
    )
    model.train(X, y, iterations=args.iterations, temperature_start=args.temp_start, temperature_end=args.temp_end)
    model.save(args.output)
    print(f"Saved trained model to {args.output}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    model = MTNCLNetwork.load(args.model)
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    preds = model.predict(inputs)
    Path(args.output).write_text(json.dumps(preds, indent=2), encoding="utf-8")
    print(f"Wrote predictions to {args.output}")
    return 0


def cmd_netlist(args: argparse.Namespace) -> int:
    model = MTNCLNetwork.load(args.model)
    verilog = model.to_verilog(module_name=args.module_name)
    Path(args.output).write_text(verilog, encoding="utf-8")
    print(f"Wrote Verilog netlist to {args.output}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    model = MTNCLNetwork.load(args.model)
    dot = model.to_dot(graph_name=args.graph_name)
    Path(args.dot_output).write_text(dot, encoding="utf-8")
    print(f"Wrote DOT graph to {args.dot_output}")

    if args.image_output:
        dot_bin = shutil.which("dot")
        if not dot_bin:
            print("Graphviz 'dot' not found; skipping image render.")
            return 0
        cmd = [dot_bin, f"-T{args.image_format}", args.dot_output, "-o", args.image_output]
        subprocess.run(cmd, check=True)
        print(f"Wrote graph image to {args.image_output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtncl", description="MTNCL training and inference CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a model from JSON dataset")
    train.add_argument("--data", required=True, help="Path to dataset JSON with keys: X, y")
    train.add_argument("--output", required=True, help="Path to write trained model JSON")
    train.add_argument("--num-inputs", type=int, required=True)
    train.add_argument("--num-outputs", type=int, required=True)
    train.add_argument("--hidden-layers", type=int, nargs="*", default=[4])
    train.add_argument("--iterations", type=int, default=500)
    train.add_argument("--temp-start", type=float, default=0.8)
    train.add_argument("--temp-end", type=float, default=0.1)
    train.add_argument("--verbose", action="store_true")
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="Run inference from saved model + input JSON")
    infer.add_argument("--model", required=True, help="Path to model JSON")
    infer.add_argument("--inputs", required=True, help="Path to JSON array of input rows")
    infer.add_argument("--output", required=True, help="Path to write predictions JSON")
    infer.set_defaults(func=cmd_infer)

    netlist = sub.add_parser("netlist", help="Export model JSON to Verilog netlist")
    netlist.add_argument("--model", required=True, help="Path to model JSON")
    netlist.add_argument("--output", required=True, help="Path to write .v file")
    netlist.add_argument("--module-name", default="mtncl_net", help="Verilog module name")
    netlist.set_defaults(func=cmd_netlist)

    visualize = sub.add_parser("visualize", help="Export model graph to DOT (and optional image via Graphviz)")
    visualize.add_argument("--model", required=True, help="Path to model JSON")
    visualize.add_argument("--dot-output", required=True, help="Path to write .dot file")
    visualize.add_argument("--graph-name", default="mtncl_net", help="DOT graph name")
    visualize.add_argument("--image-output", help="Optional path to image file (e.g., net.png, net.svg)")
    visualize.add_argument("--image-format", default="png", choices=["png", "svg", "pdf"], help="Graphviz output format")
    visualize.set_defaults(func=cmd_visualize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
