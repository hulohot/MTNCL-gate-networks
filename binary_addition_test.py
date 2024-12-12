"""
Test case: 2-bit Binary Addition
This module demonstrates using the MTNCL Neural Network framework
to implement a 2-bit binary adder.
"""

from mtncl_framework import MTNCLNetwork
from typing import List, Tuple

def generate_binary_addition_data() -> Tuple[List[List[float]], List[List[float]]]:
    """Generate training data for 2-bit binary addition"""
    X = []  # Inputs: [a1, a0, b1, b0]
    y = []  # Outputs: [s2, s1, s0] (sum with carry)
    
    # Generate all possible 2-bit number combinations
    for a in range(4):  # 2-bit numbers: 00, 01, 10, 11
        for b in range(4):
            # Convert to binary and pad to 2 bits
            a_bits = [int(x) for x in format(a, '02b')]
            b_bits = [int(x) for x in format(b, '02b')]
            
            # Calculate sum
            sum_val = a + b
            sum_bits = [int(x) for x in format(sum_val, '03b')]  # 3 bits for sum (including carry)
            
            X.append(a_bits + b_bits)  # Concatenate input bits
            y.append(sum_bits)
    
    return X, y

def main():
    # Generate binary addition training data
    X, y = generate_binary_addition_data()
    
    # Create network for binary addition (4 inputs, 3 outputs)
    network = MTNCLNetwork(num_inputs=4, num_outputs=3, hidden_layers=[8, 6])
    
    print("Training network for binary addition...")
    print("Input format: [a1, a0, b1, b0]")
    print("Output format: [s2, s1, s0] (sum with carry)")
    network.train(X, y, iterations=10000)
    
    print("\nFinal testing:")
    for x_i, y_i in zip(X, y):
        output = network.forward(x_i)
        print(f"Input: {x_i} ({x_i[0]*2 + x_i[1]} + {x_i[2]*2 + x_i[3]})")
        print(f"Expected: {y_i} = {y_i[0]*4 + y_i[1]*2 + y_i[2]}")
        print(f"Output: {[round(o) for o in output]}")
        print()
    
    # Generate VHDL netlist
    print("\nGenerating VHDL netlist...")
    network.generate_vhdl("binary_adder.vhdl")
    print("VHDL netlist generated as 'binary_adder.vhdl'")

if __name__ == "__main__":
    main() 