"""
XOR Example
This example demonstrates using the MTNCL Neural Network framework
to implement a XOR gate using MTNCL gates.

The XOR truth table:
A B | Output
0 0 | 0
0 1 | 1
1 0 | 1
1 1 | 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mtncl_nn import MTNCLNetwork
from typing import List, Tuple
import json
from datetime import datetime

def generate_xor_data() -> Tuple[List[List[float]], List[List[float]]]:
    """Generate XOR training data"""
    X = [
        [0, 0],  # Case 1: 0 XOR 0 = 0
        [0, 1],  # Case 2: 0 XOR 1 = 1
        [1, 0],  # Case 3: 1 XOR 0 = 1
        [1, 1],  # Case 4: 1 XOR 1 = 0
    ]
    
    y = [
        [0],  # Case 1 output
        [1],  # Case 2 output
        [1],  # Case 3 output
        [0],  # Case 4 output
    ]
    
    return X, y

def print_truth_table(network: MTNCLNetwork):
    """Print the truth table with network outputs"""
    print("\nXOR Truth Table:")
    print("=" * 30)
    print("A B | Expected | Output")
    print("-" * 30)
    
    X, y = generate_xor_data()
    correct = 0
    
    for inputs, expected in zip(X, y):
        output = network.forward(inputs)
        is_correct = (output[0] >= 0.5) == (expected[0] >= 0.5)
        if is_correct:
            correct += 1
        
        print(f"{inputs[0]} {inputs[1]} |    {expected[0]}    |  {output[0]:.3f}  {'✓' if is_correct else '✗'}")
    
    print("-" * 30)
    accuracy = correct / len(X) * 100
    print(f"Accuracy: {accuracy:.1f}%")
    return accuracy

def save_results(network: MTNCLNetwork, accuracy: float, timestamp: str):
    """Save network configuration and results"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Save results
    results = {
        'accuracy': accuracy,
        'training_history': network.training_history,
        'network_config': network_config
    }
    
    with open(os.path.join(output_dir, 'xor_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train and test XOR operation using MTNCL gates')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate XOR data
    X, y = generate_xor_data()
    
    # Create a minimal network for XOR
    network = MTNCLNetwork(
        num_inputs=2,
        num_outputs=1,
        hidden_layers=[4, 2],  # Small network is sufficient for XOR
        verbose=args.verbose
    )
    
    if args.verbose:
        print("Training network for XOR operation...")
        print("\nTraining Data:")
        print("Input: A B | Output")
        for x_i, y_i in zip(X, y):
            print(f"      {x_i[0]} {x_i[1]} |   {y_i[0]}")
    else:
        print("Training network for XOR operation...")
    
    # Train with focused parameters for XOR
    network.train(
        X, y,
        iterations=5000,
        temperature_start=0.8,
        temperature_end=0.1
    )
    
    # Test and print results
    accuracy = print_truth_table(network)
    
    # Save results
    save_results(network, accuracy, timestamp)
    print(f"\nResults saved in examples/outputs/{timestamp}/")
    
    # Print network structure if verbose
    if args.verbose:
        print("\nNetwork Structure:")
        print("=" * 50)
        for layer_idx, layer in enumerate(network.gates):
            print(f"\nLayer {layer_idx}:")
            for gate in layer:
                gate_type = gate.gate_type.name if gate.gate_type else "None"
                print(f"  {gate.name}: Type={gate_type}, Inputs={[g.name for g in gate.inputs]}")

if __name__ == "__main__":
    main() 