# MTNCL Neural Network Framework

Neural network framework built from Multi-Threshold Null Convention Logic (MTNCL) gates.

## What’s included

- MTNCL gate primitives (`GateType`, `MTNCLGate`)
- Network + simulated annealing training (`MTNCLNetwork`)
- Model save/load to JSON
- CLI for training and inference
- Test suite (`pytest`)
- GitHub Actions CI for PR/push validation

## Install

```bash
pip install -e .[dev]
```

## Run tests

```bash
pytest
```

## Toy models

### XOR (MTNCL)

```bash
python examples/toy_models/xor_mtncl.py --iterations 2000
# writes examples/outputs/toy_models/xor.json and xor.v
```

### MNIST (toy subset, MTNCL)

```bash
pip install -r requirements.txt
python examples/toy_models/mnist_mtncl.py --classes 0,1 --train-per-class 30 --test-per-class 10 --iterations 500
# writes examples/outputs/toy_models/mnist.json and mnist.v
```

### Export a Verilog netlist from any saved model

```bash
python -m mtncl_nn.cli netlist --model model.json --output model.v --module-name my_mtncl_net
```

## Python usage

```python
from mtncl_nn import MTNCLNetwork

X = [[0,0], [0,1], [1,0], [1,1]]
y = [[1,0], [0,1], [0,1], [1,0]]

net = MTNCLNetwork(num_inputs=2, num_outputs=2, hidden_layers=[3])
net.train(X, y, iterations=200)

pred = net.forward([1, 0])
net.save("model.json")
```

## CLI: training and inference

### 1) Prepare dataset JSON

`dataset.json`:

```json
{
  "X": [[0,0], [0,1], [1,0], [1,1]],
  "y": [[1,0], [0,1], [0,1], [1,0]]
}
```

### 2) Train

```bash
python -m mtncl_nn.cli train \
  --data dataset.json \
  --output model.json \
  --num-inputs 2 \
  --num-outputs 2 \
  --hidden-layers 3 \
  --iterations 500
```

### 3) Infer

`inputs.json`:

```json
[[0,0], [1,1]]
```

```bash
python -m mtncl_nn.cli infer \
  --model model.json \
  --inputs inputs.json \
  --output predictions.json
```

## Project layout

```
src/mtncl_nn/
  core/network.py
  gates/gate.py
  gates/gate_types.py
  cli.py
tests/
.github/workflows/ci.yml
```
