"""
Pattern Recognition Example
This example demonstrates using the MTNCL Neural Network framework
to recognize simple patterns in 3x3 binary images.

Patterns to recognize:
- Horizontal lines (3 patterns)
- Vertical lines (3 patterns)
- Diagonal lines (2 patterns)
- Empty/Noise (random patterns)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mtncl_nn import MTNCLNetwork
from typing import List, Tuple
import random
import json
from datetime import datetime

# Define basic patterns
HORIZONTAL_LINES = [
    [1, 1, 1,  # Top row
     0, 0, 0,
     0, 0, 0],
    
    [0, 0, 0,  # Middle row
     1, 1, 1,
     0, 0, 0],
    
    [0, 0, 0,  # Bottom row
     0, 0, 0,
     1, 1, 1]
]

VERTICAL_LINES = [
    [1, 0, 0,  # Left column
     1, 0, 0,
     1, 0, 0],
    
    [0, 1, 0,  # Middle column
     0, 1, 0,
     0, 1, 0],
    
    [0, 0, 1,  # Right column
     0, 0, 1,
     0, 0, 1]
]

DIAGONAL_LINES = [
    [1, 0, 0,  # Main diagonal
     0, 1, 0,
     0, 0, 1],
    
    [0, 0, 1,  # Anti-diagonal
     0, 1, 0,
     1, 0, 0]
]

def generate_random_pattern() -> List[float]:
    """Generate a random pattern with 3 pixels set"""
    pattern = [0] * 9
    indices = random.sample(range(9), 3)  # Randomly select 3 positions
    for idx in indices:
        pattern[idx] = 1
    return pattern

def generate_training_data(num_random_patterns: int = 20) -> Tuple[List[List[float]], List[List[float]]]:
    """Generate training data for pattern recognition"""
    X = []  # 9 inputs (3x3 grid)
    y = []  # 4 outputs (pattern type)
    
    # Add patterns with noise
    for pattern_list, output_encoding in [
        (HORIZONTAL_LINES, [1, 0, 0, 0]),
        (VERTICAL_LINES, [0, 1, 0, 0]),
        (DIAGONAL_LINES, [0, 0, 1, 0])
    ]:
        for pattern in pattern_list:
            # Add original pattern
            X.append(pattern)
            y.append(output_encoding)
            
            # Add noisy versions
            for _ in range(2):
                noisy_pattern = pattern.copy()
                num_flips = random.randint(1, 2)
                zero_indices = [i for i, x in enumerate(pattern) if x == 0]
                flip_indices = random.sample(zero_indices, num_flips)
                for idx in flip_indices:
                    noisy_pattern[idx] = 1
                X.append(noisy_pattern)
                y.append(output_encoding)
    
    # Add random patterns
    for _ in range(num_random_patterns):
        X.append(generate_random_pattern())
        y.append([0, 0, 0, 1])
    
    return X, y

def print_pattern(pattern: List[float]):
    """Print a 3x3 pattern in a readable format"""
    for i in range(0, 9, 3):
        print(" ".join("█" if x >= 0.5 else "·" for x in pattern[i:i+3]))

def print_classification_results(output: List[float], expected: List[float], pattern_types: List[str]):
    """Print classification results in a clear format"""
    print("\nClassification Results:")
    print("-" * 50)
    max_prob = max(output)
    max_class = pattern_types[output.index(max_prob)]
    expected_class = pattern_types[expected.index(1)] if 1 in expected else "Unknown"
    
    # Calculate bar lengths (max 20 chars)
    max_bar_length = 20
    bar_lengths = [int(prob * max_bar_length) for prob in output]
    
    # Print header
    print(f"{'Class':10} {'Actual':6} {'Expected':8} {'Probability Bar'}")
    print("-" * 50)
    
    # Print each class with probability and bar
    for class_name, prob, expected_val, bar_length in zip(pattern_types, output, expected, bar_lengths):
        bar = "█" * bar_length + "·" * (max_bar_length - bar_length)
        is_predicted = class_name == max_class
        is_expected = expected_val == 1
        
        # Create markers for predicted and expected
        markers = []
        if is_predicted:
            markers.append("P")  # Predicted
        if is_expected:
            markers.append("E")  # Expected
        marker_str = f"[{','.join(markers)}]" if markers else ""
        
        print(f"{class_name:10} {expected_val:6.0f}  {prob:8.3f} |{bar}| {marker_str}")
    
    print("-" * 50)
    if max_class == expected_class:
        print("✓ Correct classification")
    else:
        print(f"✗ Incorrect classification (expected {expected_class})")

def save_results(network: MTNCLNetwork, test_results: List[dict], timestamp: str):
    """Save network configuration and test results"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(network.training_history, f, indent=2)
    
    # Save network configuration
    network_config = {
        'num_layers': len(network.gates),
        'layer_sizes': [len(layer) for layer in network.gates],
        'gates': [
            {
                'name': gate.name,
                'layer': gate.layer,
                'type': gate.gate_type.name if gate.gate_type else None,
                'inputs': [g.name for g in gate.inputs]
            }
            for layer in network.gates
            for gate in layer
        ]
    }
    with open(os.path.join(output_dir, 'network_config.json'), 'w') as f:
        json.dump(network_config, f, indent=2)

def main():
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate training data
    X, y = generate_training_data(num_random_patterns=20)
    
    # Create network with improved architecture
    network = MTNCLNetwork(
        num_inputs=9,
        num_outputs=4,
        hidden_layers=[36, 18, 9]  # Deeper network with gradually decreasing layer sizes
    )
    
    print("Training network for pattern recognition...")
    print("\nPattern types:")
    print("1. Horizontal lines")
    print("2. Vertical lines")
    print("3. Diagonal lines")
    print("4. Random/Noise")
    
    # Train with improved parameters
    network.train(
        X, y,
        iterations=10000,  # More iterations
        temperature_start=0.8,
        temperature_end=0.1
    )
    
    # Test with some new patterns
    print("\nTesting with new patterns...")
    pattern_types = ["Horizontal", "Vertical", "Diagonal", "Random"]
    test_results = []
    
    # Test patterns with their expected classifications
    test_cases = [
        {
            "pattern": [1, 1, 1, 0, 1, 0, 0, 0, 0],
            "expected": [1, 0, 0, 0]  # Horizontal
        },
        {
            "pattern": [1, 1, 1, 0, 1, 0, 0, 0, 1],
            "expected": [1, 0, 0, 0]  # Horizontal
        },
        {
            "pattern": [0, 1, 0, 0, 1, 0, 0, 1, 0],
            "expected": [0, 1, 0, 0]  # Vertical
        },
        {
            "pattern": [1, 0, 0, 1, 0, 0, 1, 0, 0],
            "expected": [0, 1, 0, 0]  # Vertical
        },
        {
            "pattern": [1, 0, 0, 0, 1, 0, 0, 1, 1],
            "expected": [0, 0, 1, 0]  # Diagonal
        },
        {
            "pattern": [1, 0, 1, 0, 0, 1, 1, 0, 0],
            "expected": [0, 0, 0, 1]  # Random
        }
    ]
    
    total_correct = 0
    for test_case in test_cases:
        print("\nTest Pattern:")
        print_pattern(test_case["pattern"])
        output = network.forward(test_case["pattern"])
        print_classification_results(output, test_case["expected"], pattern_types)
        
        # Track accuracy
        predicted_class = pattern_types[output.index(max(output))]
        expected_class = pattern_types[test_case["expected"].index(1)]
        is_correct = predicted_class == expected_class
        if is_correct:
            total_correct += 1
        
        # Store test result
        test_results.append({
            'pattern': test_case["pattern"],
            'expected': test_case["expected"],
            'output': output,
            'predicted_class': predicted_class,
            'expected_class': expected_class,
            'is_correct': is_correct
        })
    
    # Print overall accuracy
    print("\nOverall Test Accuracy:")
    print(f"{total_correct}/{len(test_cases)} correct ({total_correct/len(test_cases)*100:.1f}%)")
    
    # Save results
    save_results(network, test_results, timestamp)
    print(f"\nResults saved in examples/outputs/{timestamp}/")

if __name__ == "__main__":
    main() 