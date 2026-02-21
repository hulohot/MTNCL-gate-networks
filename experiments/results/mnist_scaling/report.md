# MNIST Scaling Sweep (MTNCL)
This report measures toy MNIST performance versus exported circuit size for MTNCL models.

## Setup
- Input preprocessing: 28x28 -> 7x7 average pool, then binarize (>64)
- Training algorithm: MTNCL simulated annealing
- Each run uses a small balanced subset of MNIST

## Results
| config | classes | train/test per class | hidden | iters | train acc | test acc | logic gates | total gates | netlist bytes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| xor_baseline | 2 | 0/0 | [4, 2] | 800 | 100.0% | 100.0% | 8 | 12 | 766 |
| 2class_tiny | 2 | 20/10 | [16, 8] | 400 | 95.0% | 95.0% | 26 | 77 | 3641 |
| 2class_small | 2 | 30/10 | [24, 12] | 600 | 98.3% | 100.0% | 38 | 89 | 4467 |
| 4class_tiny | 4 | 15/8 | [32, 16] | 500 | 45.0% | 43.8% | 52 | 103 | 7423 |
| 4class_small | 4 | 25/10 | [40, 20] | 700 | 47.0% | 37.5% | 64 | 115 | 12389 |

## Comparison
### Accuracy vs Problem Complexity
- **xor_baseline**: 2 classes, 100.0% test accuracy
- **2class_tiny**: 2 classes, 95.0% test accuracy
- **2class_small**: 2 classes, 100.0% test accuracy
- **4class_tiny**: 4 classes, 43.8% test accuracy
- **4class_small**: 4 classes, 37.5% test accuracy

### Circuit Size vs Problem Complexity
- **xor_baseline**: 8 logic gates, 766 bytes netlist
- **2class_tiny**: 26 logic gates, 3641 bytes netlist
- **2class_small**: 38 logic gates, 4467 bytes netlist
- **4class_tiny**: 52 logic gates, 7423 bytes netlist
- **4class_small**: 64 logic gates, 12389 bytes netlist

## Standard Reference
For comparison, typical small MLPs on full MNIST (10 classes, 60k train / 10k test):
- Small MLP (~100k params): 97-98% accuracy
- LeNet-style CNN (~50k params): 99%+ accuracy
- Digital circuit equivalent: thousands to millions of gates

These MTNCL toy models are orders of magnitude smaller and run on tiny subsets.

## Notes
- These are **toy** runs, not full MNIST benchmarks.
- Small test sets can inflate variance; compare trends, not absolute SOTA numbers.
- Artifacts (model/netlist/dot) are stored in `experiments/results/mnist_scaling/models/`.
