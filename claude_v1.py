import numpy as np
from enum import Enum
from typing import List, Tuple, Dict
from dataclasses import dataclass

class MTNCLGateType(Enum):
    # Single Output Gates
    TH12 = "TH12"   # Threshold 1 of 2 inputs
    TH13 = "TH13"   # Threshold 1 of 3 inputs
    TH14 = "TH14"   # Threshold 1 of 4 inputs
    TH22 = "TH22"   # Threshold 2 of 2 inputs
    TH23 = "TH23"   # Threshold 2 of 3 inputs
    TH23w2 = "TH23w2"  # Threshold 2 of 3 inputs with weight 2 on input
    TH24 = "TH24"   # Threshold 2 of 4 inputs
    TH24w2 = "TH24w2"  # Threshold 2 of 4 inputs with weight 2
    TH24w22 = "TH24w22"  # Threshold 2 of 4 inputs with two weight-2 inputs
    TH33 = "TH33"   # Threshold 3 of 3 inputs
    TH33w2 = "TH33w2"  # Threshold 3 of 3 inputs with weight 2
    TH34 = "TH34"   # Threshold 3 of 4 inputs
    TH34w2 = "TH34w2"  # Threshold 3 of 4 inputs with weight 2
    TH34w3 = "TH34w3"  # Threshold 3 of 4 inputs with weight 3
    TH34w22 = "TH34w22"  # Threshold 3 of 4 inputs with two weight-2 inputs
    TH44 = "TH44"   # Threshold 4 of 4 inputs
    TH44w2 = "TH44w2"  # Threshold 4 of 4 inputs with weight 2
    TH44w3 = "TH44w3"  # Threshold 4 of 4 inputs with weight 3
    TH44w22 = "TH44w22"  # Threshold 4 of 4 inputs with two weight-2 inputs
    THxor = "THxor"  # Threshold XOR gate
    THand = "THand"  # Threshold AND gate
    THor = "THor"   # Threshold OR gate

@dataclass
class GateConfig:
    input_count: int
    weight_config: List[float]
    threshold: float

class MTNCLGate:
    GATE_CONFIGS = {
        MTNCLGateType.TH12: GateConfig(2, [1, 1], 1),
        MTNCLGateType.TH13: GateConfig(3, [1, 1, 1], 1),
        MTNCLGateType.TH14: GateConfig(4, [1, 1, 1, 1], 1),
        MTNCLGateType.TH22: GateConfig(2, [1, 1], 2),
        MTNCLGateType.TH23: GateConfig(3, [1, 1, 1], 2),
        MTNCLGateType.TH23w2: GateConfig(3, [2, 1, 1], 2),
        MTNCLGateType.TH24: GateConfig(4, [1, 1, 1, 1], 2),
        MTNCLGateType.TH24w2: GateConfig(4, [2, 1, 1, 1], 2),
        MTNCLGateType.TH24w22: GateConfig(4, [2, 2, 1, 1], 2),
        MTNCLGateType.TH33: GateConfig(3, [1, 1, 1], 3),
        MTNCLGateType.TH33w2: GateConfig(3, [2, 1, 1], 3),
        MTNCLGateType.TH34: GateConfig(4, [1, 1, 1, 1], 3),
        MTNCLGateType.TH34w2: GateConfig(4, [2, 1, 1, 1], 3),
        MTNCLGateType.TH34w3: GateConfig(4, [3, 1, 1, 1], 3),
        MTNCLGateType.TH34w22: GateConfig(4, [2, 2, 1, 1], 3),
        MTNCLGateType.TH44: GateConfig(4, [1, 1, 1, 1], 4),
        MTNCLGateType.TH44w2: GateConfig(4, [2, 1, 1, 1], 4),
        MTNCLGateType.TH44w3: GateConfig(4, [3, 1, 1, 1], 4),
        MTNCLGateType.TH44w22: GateConfig(4, [2, 2, 1, 1], 4),
        MTNCLGateType.THxor: GateConfig(2, [1, 1], 1),
        MTNCLGateType.THand: GateConfig(2, [1, 1], 2),
        MTNCLGateType.THor: GateConfig(2, [1, 1], 1)
    }

    def __init__(self, gate_type: MTNCLGateType, thresholds: List[float]):
        self.gate_type = gate_type
        self.config = self.GATE_CONFIGS[gate_type]
        self.thresholds = thresholds
        self.output_history = []
        self.input_history = []
        self.weighted_history = []

    def _apply_weights(self, inputs: List[float]) -> List[float]:
        """Apply weight configuration to inputs"""
        if len(inputs) != self.config.input_count:
            raise ValueError(f"Gate {self.gate_type} expects {self.config.input_count} inputs")
        return [w * x for w, x in zip(self.config.weight_config, inputs)]

    def _threshold_function(self, weighted_sum: float) -> float:
        """Apply threshold function with learned thresholds"""
        for threshold in sorted(self.thresholds):
            if weighted_sum >= threshold:
                return 1.0
        return 0.0

    def forward(self, inputs: List[float]) -> float:
        self.input_history.append(inputs)
        weighted_inputs = self._apply_weights(inputs)
        self.weighted_history.append(weighted_inputs)

        weighted_sum = sum(weighted_inputs)
        
        if self.gate_type == MTNCLGateType.THxor:
            # Special XOR implementation
            output = 1.0 if abs(weighted_sum - self.thresholds[0]) < self.thresholds[1] else 0.0
        else:
            output = self._threshold_function(weighted_sum)
        
        self.output_history.append(output)
        return output

    def backward(self, gradient: float) -> List[float]:
        """Compute gradients for inputs using straight-through estimator"""
        weighted_inputs = self.weighted_history[-1]
        input_gradients = []
        
        for i, (w_input, weight) in enumerate(zip(weighted_inputs, self.config.weight_config)):
            # Straight-through estimator with weight scaling
            input_grad = gradient * weight
            input_gradients.append(input_grad)
        
        return input_gradients

class MTNCLLayer:
    def __init__(self, input_size: int, gate_configs: List[Tuple[MTNCLGateType, List[float]]]):
        self.gates = [MTNCLGate(gate_type, thresholds) 
                     for gate_type, thresholds in gate_configs]
        self.input_size = input_size

    def _pad_inputs(self, inputs: List[float], target_size: int) -> List[float]:
        """Pad inputs with zeros to match gate input requirements"""
        return inputs + [0.0] * (target_size - len(inputs))

    def forward(self, inputs: List[float]) -> List[float]:
        outputs = []
        for gate in self.gates:
            # Pad inputs to match gate's required input count
            padded_inputs = self._pad_inputs(inputs, gate.config.input_count)
            outputs.append(gate.forward(padded_inputs))
        return outputs

    def backward(self, gradients: List[float]) -> List[float]:
        all_input_gradients = []
        for gate, gradient in zip(self.gates, gradients):
            input_gradients = gate.backward(gradient)
            all_input_gradients.append(input_gradients)
        
        # Sum gradients for each input
        summed_gradients = [sum(grads[i] for grads in all_input_gradients)
                          for i in range(self.input_size)]
        return summed_gradients

class MTNCLNeuralNetwork:
    def __init__(self):
        # Create network using appropriate MTNCL gates for XOR
        # First layer: 2-input gates since we have 2 inputs (a, b)
        self.layers = [
            MTNCLLayer(2, [
                (MTNCLGateType.TH12, [0.3]),  # 2-input gate
                (MTNCLGateType.TH22, [0.5]),  # 2-input gate
                (MTNCLGateType.THand, [0.4])  # 2-input gate
            ]),
            MTNCLLayer(3, [
                (MTNCLGateType.TH13, [0.4]),  # 3-input gate for combining previous layer
                (MTNCLGateType.TH23, [0.3, 0.5])  # Final XOR decision
            ])
        ]

    def forward(self, x: List[float]) -> float:
        current_output = x
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output[-1]  # Return final output

    def backward(self, gradient: float) -> None:
        current_gradient = [0] * (len(self.layers[-1].gates) - 1) + [gradient]
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)

    def update_thresholds(self, learning_rate: float) -> None:
        for layer in self.layers:
            for gate in layer.gates:
                for i in range(len(gate.thresholds)):
                    # Compute gradient estimate based on output history
                    gradient_estimate = np.mean(gate.output_history) - 0.5
                    gate.thresholds[i] -= learning_rate * gradient_estimate
                    # Ensure thresholds stay in reasonable range
                    gate.thresholds[i] = max(0.1, min(0.9, gate.thresholds[i]))
                
                gate.output_history = []
                gate.input_history = []
                gate.weighted_history = []

def generate_vhdl(network: MTNCLNeuralNetwork, filename: str) -> None:
    with open(filename, 'w') as f:
        # Write VHDL entity and architecture
        f.write("""
library IEEE;
use IEEE.std_logic_1164.all;

entity MTNCLNeuralNetwork is
    port (
        a, b: in std_logic;
        result: out std_logic;
        reset, sleep: in std_logic
    );
end MTNCLNeuralNetwork;

architecture behavioral of MTNCLNeuralNetwork is
""")
        
        # Write component declarations for all MTNCL gates
        for gate_type in MTNCLGateType:
            config = MTNCLGate.GATE_CONFIGS[gate_type]
            port_list = ", ".join([f"a{i+1}" for i in range(config.input_count)])
            
            f.write(f"""
    component {gate_type.value}
        port (
            {port_list}: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;
""")

        # Write signals for interconnections
        f.write("\n    -- Internal signals\n")
        layer_counts = [len(layer.gates) for layer in network.layers]
        for i, count in enumerate(layer_counts):
            for j in range(count):
                f.write(f"    signal layer{i+1}_out{j+1}: std_logic;\n")

        f.write("\nbegin\n")

        # Instantiate gates for each layer
        for layer_idx, layer in enumerate(network.layers):
            f.write(f"\n    -- Layer {layer_idx + 1}\n")
            for gate_idx, gate in enumerate(layer.gates):
                port_map = []
                if layer_idx == 0:
                    # First layer connects to inputs
                    port_map.extend(["a", "b"])
                else:
                    # Other layers connect to previous layer outputs
                    prev_outputs = [f"layer{layer_idx}_out{i+1}" 
                                  for i in range(len(network.layers[layer_idx-1].gates))]
                    port_map.extend(prev_outputs)
                
                # Add remaining ports
                while len(port_map) < gate.config.input_count:
                    port_map.append("'0'")
                
                # Write gate instance
                f.write(f"""
    gate_l{layer_idx+1}_{gate_idx+1}: {gate.gate_type.value}
        port map (
            {', '.join(f'a{i+1} => {port}' for i, port in enumerate(port_map))},
            z => layer{layer_idx+1}_out{gate_idx+1},
            sleep => sleep
        );
""")

        # Connect final output
        f.write("\n    result <= layer2_out2;  -- Final output\n")
        f.write("end behavioral;\n")

def train_network():
    network = MTNCLNeuralNetwork()
    learning_rate = 0.01
    epochs = 1000
    
    # XOR training data
    training_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in training_data:
            # Forward pass
            output = network.forward(inputs)
            
            # Calculate loss
            loss = (output - target) ** 2
            total_loss += loss
            
            # Backward pass
            gradient = 2 * (output - target)
            network.backward(gradient)
            
            # Update thresholds
            network.update_thresholds(learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/4:.4f}")
    
    # Generate VHDL
    generate_vhdl(network, "mtncl_network.vhdl")
    return network

if __name__ == "__main__":
    network = train_network()
    
    # Test the trained network
    test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    print("\nTesting trained network:")
    for inputs in test_inputs:
        output = network.forward(inputs)
        print(f"Input: {inputs}, Output: {output:.3f}")