# MTNCL Neural Network Framework

A framework for building neural networks using Multi-Threshold Null Convention Logic (MTNCL) gates. This project implements neural networks using MTNCL gates instead of traditional Boolean gates, making it suitable for asynchronous circuit implementations.

## Project Structure

```
mtncl-neural-network-gen/
├── src/
│   └── mtncl_nn/
│       ├── core/           # Core network implementation
│       ├── gates/          # Gate definitions and implementations
│       └── utils/          # Utility functions
├── examples/
│   ├── pattern_recognition/  # Pattern recognition example
│   ├── xor/                # XOR operation example
│   └── outputs/            # Example outputs and results
├── tests/                  # Test cases
└── docs/                   # Documentation
```

## Features

- Implementation of various MTNCL gate types
- Neural network framework using MTNCL gates
- Training using simulated annealing
- Binary output classification
- Detailed performance analysis and debugging
- Example implementations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mtncl-neural-network-gen.git
cd mtncl-neural-network-gen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.mtncl_nn import MTNCLNetwork

# Create a network
network = MTNCLNetwork(
    num_inputs=9,
    num_outputs=4,
    hidden_layers=[36, 18, 9]
)

# Train the network
network.train(X, y, iterations=10000)

# Use the network
output = network.forward(input_data)
```

### Running Examples

1. Pattern Recognition Example:
```bash
cd examples/pattern_recognition
python pattern_recognition.py
```

2. XOR Operation Example:
```bash
cd examples/xor
python xor_example.py
```

## Examples

### Pattern Recognition

The pattern recognition example demonstrates using the MTNCL neural network to recognize simple patterns in 3x3 binary images:
- Horizontal lines
- Vertical lines
- Diagonal lines
- Random patterns

### XOR Operation

The XOR example demonstrates implementing a basic logic operation using MTNCL gates:
- Simple 2-input, 1-output network
- Complete truth table verification
- Network structure visualization
- Detailed performance analysis

Results for all examples are saved in the `examples/outputs/` directory with timestamps.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- List any acknowledgments or references here
