# Sparse Neural Networks with MTNCL Threshold Gates

## Working title
Sparse Asynchronous Logic Networks: MTNCL Threshold Gates for Efficient Learned Inference

## Motivation
Recent differentiable logic gate networks show that logic-based learned models can dramatically reduce inference cost and model size while preserving useful accuracy. MTNCL threshold gates are a natural fit for this direction because they:
- encode weighted-threshold behavior directly,
- support asynchronous handshaking and low switching activity,
- can be composed into sparse logic trees and pipelines.

Reference paper discussed:
- Convolutional Differentiable Logic Gate Networks (arXiv:2411.04732, NeurIPS 2024 Oral)

## Repository work completed this week (MTNCL-gate-networks)
Commit timeline (last 7 days):
1. `a10aa1e` (2026-02-20)
   - Added MTNCL toy XOR/MNIST examples and training fixes.
2. `61e1a14` (2026-02-20)
   - Added Verilog netlist export for trained MTNCL models.
3. `96eff7a` (2026-02-20)
   - Added netlist visualization, Icarus Verilog smoke test, and toy artifacts.

Current project structure highlights:
- Training/examples:
  - `examples/toy_models/xor_mtncl.py`
  - `examples/toy_models/mnist_mtncl.py`
  - `examples/pattern_recognition/pattern_recognition.py`
- Experiments:
  - `experiments/mnist_scaling_sweep.py`
- Core package:
  - `src/mtncl_nn/*`
- Tests:
  - `tests/test_network.py`
  - `tests/test_gate_types.py`
  - `tests/test_cli.py`
  - `tests/test_iverilog_smoke.py`

## Short paper outline (target: 4–6 pages)
1. Introduction
   - Why sparse inference and logic-native learning matter.
   - Why MTNCL threshold gates are an appealing primitive.
2. Background
   - Differentiable logic gate networks (from arXiv:2411.04732).
   - MTNCL threshold logic and asynchronous handshake assumptions.
3. Proposed Method
   - Mapping learned sparse logic to MTNCL threshold-gate networks.
   - Optional training constraints: fan-in caps, gate sparsity penalties.
   - Optional asynchronous stage partitioning and completion signaling.
4. Tool Flow
   - Train in Python (`mtncl_nn`) → export Verilog netlist → smoke test with Icarus.
   - Discuss path to ASIC flow (synthesis/P&R) for timing/power measurements.
5. Preliminary Results
   - XOR and toy MNIST trends.
   - Netlist size and sparsity indicators.
6. Discussion
   - Throughput/latency trade-offs, robustness, and scaling limits.
7. Conclusion and future work.

## High-value future work
1. Sparsity-aware training for MTNCL constraints
   - Add explicit L0/L1 regularization and gate fan-in penalties.
2. CIFAR-10-scale logic experiments
   - Reproduce a small conv logic baseline and compare MTNCL variants.
3. Hardware-aware co-optimization
   - Include area/delay/power differentiable proxies in training.
4. Asynchronous pipeline experiments
   - Partition network into handshake stages; measure cycle time and switching activity.
5. Formal/equivalence checks
   - Verify trained-exported logic consistency and robustness under quantization.
6. PVT and robustness study
   - Evaluate threshold sensitivity and completion behavior across corners.

## Immediate next steps (1–2 weeks)
- Add metric logging for gate count, average fan-in, and depth per model.
- Add benchmark script for XOR + MNIST + scaling sweep into a single report table.
- Add one reproducible “paper figure” script (accuracy vs gate count).
- Draft 1-page extended abstract from this note.
