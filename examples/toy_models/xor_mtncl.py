"""Toy XOR classifier using MTNCLNetwork.

Run:
  python examples/toy_models/xor_mtncl.py --iterations 2000
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT, "src"))

from mtncl_nn import MTNCLNetwork


def xor_dataset():
    # Two-class one-hot targets (required by current MTNCLNetwork one-hot output behavior)
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [
        [1, 0],  # XOR=0
        [0, 1],  # XOR=1
        [0, 1],
        [1, 0],
    ]
    return X, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    X, y = xor_dataset()

    net = MTNCLNetwork(
        num_inputs=2,
        num_outputs=2,
        hidden_layers=[4, 2],
        verbose=args.verbose,
    )
    net.train(X, y, iterations=args.iterations)

    print("\nXOR results")
    print("A B | pred_class | expected_class")
    for x_i, y_i in zip(X, y):
        pred = net.forward(x_i)
        pred_class = pred.index(1.0)
        exp_class = y_i.index(1.0)
        print(f"{x_i[0]} {x_i[1]} |     {pred_class}      |      {exp_class}")


if __name__ == "__main__":
    main()
