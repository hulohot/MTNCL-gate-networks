"""
MTNCL Neural Network Framework
This module provides the core functionality for building and training neural networks
using Multi-Threshold Null Convention Logic (MTNCL) gates.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum, auto
import random
from itertools import product
import time

class GateType(Enum):
    """All available MTNCL gates with their functions"""
    ALWAYS_ONE = auto()  # Always outputs 1
    ALWAYS_ZERO = auto() # Always outputs 0
    TH12 = auto()      # A + B
    TH22 = auto()      # AB
    TH13 = auto()      # A + B + C
    TH23 = auto()      # AB + AC + BC
    TH33 = auto()      # ABC
    TH23w2 = auto()    # A + BC (B has weight 2)
    TH33w2 = auto()    # AB + AC (A has weight 2)
    TH14 = auto()      # A + B + C + D
    TH24 = auto()      # AB + AC + AD + BC + BD + CD
    TH34 = auto()      # ABC + ABD + ACD + BCD
    TH44 = auto()      # ABCD
    TH24w2 = auto()    # A + BC + BD + CD (B has weight 2)
    TH34w2 = auto()    # AB + AC + AD + BCD (A has weight 2)
    TH44w2 = auto()    # ABC + ABD + ACD (A has weight 2)
    TH34w3 = auto()    # A + BCD (A has weight 3)
    TH44w3 = auto()    # AB + AC + AD (A has weight 3)
    TH24w22 = auto()   # A + B + CD (C,D have weight 2)
    TH34w22 = auto()   # AB + AC + AD + BC + BD (A,B have weight 2)
    TH44w22 = auto()   # AB + ACD + BCD (A,B have weight 2)
    TH54w22 = auto()   # ABC + ABD (A,B have weight 2)
    TH34w32 = auto()   # A + BC + BD (A has weight 3, B has weight 2)
    TH54w32 = auto()   # AB + ACD (A has weight 3, B has weight 2)
    TH44w322 = auto()  # AB + AC + AD + BC (A has weight 3, B has weight 2, C has weight 2)
    TH54w322 = auto()  # AB + AC + BCD (A has weight 3, B has weight 2, C has weight 2)
    THxor0 = auto()    # AB + CD (specialized XOR gate)
    THand0 = auto()    # AB + BC + AD (specialized AND gate)
    TH24comp = auto()  # AC + BC + AD + BD (specialized comparator)

    def compute_output(self, inputs: List[float]) -> float:
        """Compute the output of this gate type given its inputs"""
        if self == GateType.ALWAYS_ONE:
            return 1.0
        elif self == GateType.ALWAYS_ZERO:
            return 0.0
        
        # Determine required number of inputs based on gate type
        if self in {GateType.THxor0, GateType.THand0, GateType.TH12, GateType.TH22}:
            num_required = 2
        elif self in {GateType.TH13, GateType.TH23, GateType.TH33, GateType.TH23w2, GateType.TH33w2}:
            num_required = 3
        elif self in {GateType.TH14, GateType.TH24, GateType.TH34, GateType.TH44}:
            num_required = 4
        else:
            # For other gates, try to parse the number from the name
            try:
                num_required = int(self.name[2]) if len(self.name) >= 3 else 2
            except ValueError:
                num_required = 2  # Default to 2 inputs if parsing fails
        
        # Ensure we have enough inputs by padding with zeros if necessary
        padded_inputs = inputs + [0.0] * (num_required - len(inputs))
        
        # Convert inputs to boolean for easier logic computation
        bool_inputs = [i >= 0.5 for i in padded_inputs]
        
        if self == GateType.TH12:
            return float(any(bool_inputs[:2]))  # A + B
        elif self == GateType.TH22:
            return float(all(bool_inputs[:2]))  # AB
        elif self == GateType.TH13:
            return float(sum(bool_inputs[:3]) >= 1)  # A + B + C
        elif self == GateType.TH23:
            return float(sum(bool_inputs[:3]) >= 2)  # AB + AC + BC
        elif self == GateType.TH33:
            return float(sum(bool_inputs[:3]) == 3)  # ABC
        elif self == GateType.THxor0:
            return float(bool_inputs[0] ^ bool_inputs[1])  # XOR
        elif self == GateType.THand0:
            return float(bool_inputs[0] and bool_inputs[1])  # AND
        elif self == GateType.TH23w2:
            # A + BC (B has weight 2)
            A, B = bool_inputs[:2]
            C = any(bool_inputs[2:])  # Consider any additional inputs
            return float(A or (B and C))
        elif self == GateType.TH33w2:
            # AB + AC (A has weight 2)
            A, B = bool_inputs[:2]
            C = any(bool_inputs[2:])  # Consider any additional inputs
            return float((A and B) or (A and C))
        
        # For other gates, implement a basic threshold check
        threshold = num_required // 2  # Default threshold is half of required inputs
        return float(sum(bool_inputs[:num_required]) >= threshold)

    @staticmethod
    def get_compatible_gates(num_inputs: int) -> Set['GateType']:
        """Get all gate types compatible with the given number of inputs"""
        gates = {GateType.ALWAYS_ONE, GateType.ALWAYS_ZERO}
        
        if num_inputs >= 2:
            gates.update({GateType.TH12, GateType.TH22, GateType.THxor0, GateType.THand0})
        if num_inputs >= 3:
            gates.update({GateType.TH13, GateType.TH23, GateType.TH33, GateType.TH23w2, GateType.TH33w2})
        if num_inputs >= 4:
            gates.update({
                GateType.TH14, GateType.TH24, GateType.TH34, GateType.TH44,
                GateType.TH24w2, GateType.TH34w2, GateType.TH44w2,
                GateType.TH34w3, GateType.TH44w3, GateType.TH24w22,
                GateType.TH24comp
            })
        return gates

@dataclass
class MTNCLGate:
    """Represents a single MTNCL gate in the network"""
    name: str
    layer: int
    inputs: List['MTNCLGate']
    gate_type: Optional[GateType] = None
    value: float = 0.0
    
    def __hash__(self):
        """Make gate hashable based on its name (which is unique)"""
        return hash(self.name)
    
    def __eq__(self, other):
        """Define equality based on gate name"""
        if not isinstance(other, MTNCLGate):
            return False
        return self.name == other.name
    
    def forward(self) -> float:
        """Compute the output value of this gate"""
        if not self.inputs:
            return self.value  # Input gate
        if self.gate_type is None:
            return 0.0  # Gate type not yet assigned
        
        # Get input values through forward propagation
        input_values = [gate.forward() for gate in self.inputs]
        self.value = self.gate_type.compute_output(input_values)
        return self.value

class MTNCLNetwork:
    """Neural network implementation using MTNCL gates"""
    
    def __init__(self, num_inputs: int, num_outputs: int, hidden_layers: List[int]):
        """Initialize network with specified layer sizes"""
        self.gates: List[List[MTNCLGate]] = []
        self.gate_counter = 0
        self._gate_options_cache = {}  # Cache for gate options
        self.training_history = {
            'errors': [],
            'accuracies': [],
            'best_configs': [],
            'layer_stats': []
        }
        
        # Layer 0: Input gates and constants
        input_layer = [
            MTNCLGate(name=self._create_gate_name("input"), layer=0, inputs=[])
            for _ in range(num_inputs)
        ]
        constant_gates = [
            MTNCLGate(name=self._create_gate_name("const"), layer=0, inputs=[], 
                     gate_type=GateType.ALWAYS_ONE),
            MTNCLGate(name=self._create_gate_name("const"), layer=0, inputs=[],
                     gate_type=GateType.ALWAYS_ZERO)
        ]
        self.gates.append(input_layer + constant_gates)
        
        # Create hidden layers
        for layer_idx, size in enumerate(hidden_layers, 1):
            layer = []
            for _ in range(size):
                available_inputs = [g for l in self.gates for g in l]
                gate = MTNCLGate(
                    name=self._create_gate_name(f"gate_l{layer_idx}"),
                    layer=layer_idx,
                    inputs=available_inputs
                )
                layer.append(gate)
            self.gates.append(layer)
        
        # Create output layer
        output_layer = []
        for i in range(num_outputs):
            available_inputs = [g for l in self.gates for g in l]
            gate = MTNCLGate(
                name=self._create_gate_name(f"output_{i}"),
                layer=len(hidden_layers) + 1,
                inputs=available_inputs
            )
            output_layer.append(gate)
        self.gates.append(output_layer)
        
        self.output_gates = output_layer
    
    def _create_gate_name(self, prefix: str = "gate") -> str:
        self.gate_counter += 1
        return f"{prefix}_{self.gate_counter}"
    
    def get_gate_options(self, gate: MTNCLGate) -> Set[GateType]:
        """Get compatible gates with performance optimization"""
        num_inputs = len(gate.inputs)
        if num_inputs not in self._gate_options_cache:
            self._gate_options_cache[num_inputs] = GateType.get_compatible_gates(num_inputs)
        return self._gate_options_cache[num_inputs]
    
    def _smart_gate_init(self, temperature: float) -> None:
        """Initialize gates with problem-specific preferences"""
        preferred_gates = {
            'arithmetic': [GateType.THxor0, GateType.THand0, GateType.TH23w2],
            'logic': [GateType.TH12, GateType.TH22, GateType.TH23],
            'comparison': [GateType.TH24comp, GateType.TH23w2]
        }
        
        for layer in self.gates[1:]:
            for gate in layer:
                if random.random() < temperature:
                    # Use preferred gates
                    compatible_preferred = [g for g in preferred_gates['arithmetic'] 
                                         if g in self.get_gate_options(gate)]
                    if compatible_preferred:
                        gate.gate_type = random.choice(compatible_preferred)
                    else:
                        gate.gate_type = random.choice(list(self.get_gate_options(gate)))
                else:
                    # Random selection
                    gate.gate_type = random.choice(list(self.get_gate_options(gate)))
    
    def _print_debug_info(self, iteration: int, error: float, accuracy: float, X: List[List[float]], y: List[List[float]]):
        """Print detailed debug information"""
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}")
        print(f"Error: {error:.4f}, Accuracy: {accuracy*100:.2f}%")
        
        # Print layer-by-layer analysis
        # print("\nLayer Analysis:")
        layer_stats = []
        for layer_idx, layer in enumerate(self.gates[1:], 1):
            # print(f"\nLayer {layer_idx}:")
            layer_stat = {'layer': layer_idx, 'gates': []}
            for gate in layer:
                # Get gate's output for all inputs
                outputs = []
                for x in X:
                    self.forward(x)
                    outputs.append(gate.value)
                
                # Calculate gate statistics
                avg_output = sum(outputs) / len(outputs)
                always_high = all(o > 0.5 for o in outputs)
                always_low = all(o < 0.5 for o in outputs)
                switching = sum(1 for i in range(1, len(outputs)) if outputs[i] != outputs[i-1])
                
                gate_stat = {
                    'name': gate.name,
                    'type': gate.gate_type.name,
                    'avg_output': avg_output,
                    'stuck': 'HIGH' if always_high else 'LOW' if always_low else 'NO',
                    'switching_rate': switching / (len(outputs) - 1)
                }
                layer_stat['gates'].append(gate_stat)
                
                # print(f"  {gate.name} ({gate.gate_type.name}):")
                # print(f"    Avg Output: {avg_output:.2f}")
                # print(f"    {'[STUCK HIGH]' if always_high else '[STUCK LOW]' if always_low else '[ACTIVE]'}")
                # print(f"    Switching Rate: {switching/(len(outputs)-1)*100:.1f}%")
                # print(f"    Inputs: {[g.name for g in gate.inputs]}")
            
            layer_stats.append(layer_stat)
        
        self.training_history['layer_stats'].append(layer_stats)
        
        # Print output analysis
        print("\nOutput Analysis:")
        for i, output_gate in enumerate(self.output_gates):
            outputs = []
            for x in X:
                out = self.forward(x)
                outputs.append(round(out[i]))
            
            correct = sum(1 for pred, true in zip(outputs, [yi[i] for yi in y]) if pred == true)
            print(f"\nOutput {i}:")
            print(f"  Accuracy: {correct/len(outputs)*100:.2f}%")
            print("  Last few predictions vs expected:")
            for j in range(min(5, len(outputs))):
                print(f"    {X[j]} -> Pred: {outputs[j]}, True: {y[j][i]}")
    
    def train(self, X: List[List[float]], y: List[List[float]], iterations: int = 1000,
             temperature_start: float = 0.8, temperature_end: float = 0.1) -> None:
        """Train the network using a modified simulated annealing approach"""
        best_error = float('inf')
        best_accuracy = 0.0
        best_config = None
        plateau_counter = 0
        max_plateau = 50  # Number of iterations without improvement before resetting
        
        # Initialize with smart gate selection
        self._smart_gate_init(temperature=1.0)
        error, accuracy = self.evaluate(X, y)
        
        for iteration in range(iterations):
            # Anneal temperature
            temperature = temperature_start - (temperature_start - temperature_end) * (iteration / iterations)
            
            # Store current configuration
            current_config = [(gate.gate_type, gate.inputs[:]) for layer in self.gates[1:] for gate in layer]
            
            # Perform multiple local modifications
            num_modifications = max(1, int(4 * temperature))  # More modifications at higher temperatures
            modified_gates = set()
            
            for _ in range(num_modifications):
                # Select a random non-input layer and gate
                layer_idx = random.randint(1, len(self.gates) - 1)
                gate = random.choice(self.gates[layer_idx])
                
                if gate not in modified_gates:
                    modified_gates.add(gate)
                    
                    if random.random() < 0.7:  # 70% chance to change gate type
                        # Change gate type
                        old_type = gate.gate_type
                        compatible_gates = self.get_gate_options(gate)
                        new_type = random.choice([g for g in compatible_gates if g != old_type])
                        gate.gate_type = new_type
                    else:  # 30% chance to modify inputs
                        # Modify input connections
                        available_inputs = [g for l in self.gates[:layer_idx] for g in l]
                        if len(available_inputs) > 2:  # Ensure we have enough inputs to choose from
                            num_inputs = random.randint(2, min(4, len(available_inputs)))
                            gate.inputs = random.sample(available_inputs, num_inputs)
            
            # Evaluate new configuration
            new_error, new_accuracy = self.evaluate(X, y)
            
            # Accept or reject changes based on temperature and improvement
            if new_accuracy > best_accuracy or random.random() < temperature:
                best_error = min(best_error, new_error)
                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_config = current_config
                    plateau_counter = 0
                    self._print_debug_info(iteration, new_error, new_accuracy, X, y)
            else:
                # Revert changes
                for gate, (old_type, old_inputs) in zip(
                    (g for l in self.gates[1:] for g in l),
                    current_config
                ):
                    gate.gate_type = old_type
                    gate.inputs = old_inputs
                plateau_counter += 1
            
            # Reset if stuck in plateau
            if plateau_counter >= max_plateau:
                print(f"\nReset at iteration {iteration} due to plateau")
                self._smart_gate_init(temperature=1.0)
                plateau_counter = 0
            
            # Store training history
            self.training_history['errors'].append(new_error)
            self.training_history['accuracies'].append(new_accuracy)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Error = {new_error:.4f}, Accuracy = {new_accuracy*100:.1f}%, "
                      f"Best Accuracy = {best_accuracy*100:.1f}%")
        
        # Restore best configuration
        if best_config:
            for gate, (best_type, best_inputs) in zip(
                (g for l in self.gates[1:] for g in l),
                best_config
            ):
                gate.gate_type = best_type
                gate.inputs = best_inputs
        
        print(f"\nTraining completed. Best accuracy: {best_accuracy*100:.1f}%")
    
    def evaluate(self, X: List[List[float]], y: List[List[float]]) -> Tuple[float, float]:
        """Evaluate network performance on given inputs"""
        total_error = 0.0
        correct_outputs = 0
        total_outputs = len(X)
        
        for x_i, y_i in zip(X, y):
            outputs = self.forward(x_i)
            # Since outputs are now binary, we can directly compare
            if outputs == y_i:
                correct_outputs += 1
            # Error is 1 if wrong, 0 if correct (binary case)
            total_error += 0 if outputs == y_i else 1
        
        error_rate = total_error / len(X)
        accuracy = correct_outputs / len(X)
        
        return error_rate, accuracy

    def forward(self, x: List[float]) -> List[float]:
        """Forward propagate input through the network"""
        # Set input values
        for input_gate, value in zip(self.gates[0][:len(x)], x):
            input_gate.value = value
        
        # Forward propagate through network
        raw_outputs = [gate.forward() for gate in self.output_gates]
        
        # Convert to one-hot output (only highest value is 1, rest are 0)
        binary_outputs = self._to_one_hot(raw_outputs)
        return binary_outputs
    
    def _to_one_hot(self, outputs: List[float]) -> List[float]:
        """Convert raw outputs to one-hot encoding (binary outputs)
        
        This can be implemented with MTNCL gates using a series of
        threshold gates to compare each output with others.
        """
        if not outputs:
            return []
        
        # Find the maximum output
        max_val = max(outputs)
        max_idx = outputs.index(max_val)
        
        # Set the highest output to 1, others to 0
        return [1.0 if i == max_idx else 0.0 for i in range(len(outputs))]
    
    def get_performance_stats(self, X: List[List[float]], y: List[List[float]]) -> Dict:
        """Get detailed performance statistics"""
        total_cases = len(X)
        correct_cases = 0
        bit_accuracy = 0
        confusion_matrix = {}  # For each output position
        
        # Initialize confusion matrix for each output bit
        for i in range(len(y[0])):
            confusion_matrix[i] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        
        for x_i, y_i in zip(X, y):
            outputs = self.forward(x_i)
            rounded_outputs = [round(o) for o in outputs]
            
            # Check if entire case is correct
            if rounded_outputs == y_i:
                correct_cases += 1
            
            # Update confusion matrix for each bit
            for i, (expected, actual) in enumerate(zip(y_i, rounded_outputs)):
                if expected == 1 and actual == 1:
                    confusion_matrix[i]["TP"] += 1
                elif expected == 0 and actual == 0:
                    confusion_matrix[i]["TN"] += 1
                elif expected == 0 and actual == 1:
                    confusion_matrix[i]["FP"] += 1
                else:  # expected == 1 and actual == 0
                    confusion_matrix[i]["FN"] += 1
                
                # Count correct bits
                if expected == actual:
                    bit_accuracy += 1
        
        # Calculate statistics
        case_accuracy = correct_cases / total_cases
        bit_accuracy = bit_accuracy / (total_cases * len(y[0]))
        
        # Calculate per-bit metrics
        bit_metrics = {}
        for i in range(len(y[0])):
            cm = confusion_matrix[i]
            total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
            accuracy = (cm["TP"] + cm["TN"]) / total if total > 0 else 0
            precision = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) > 0 else 0
            recall = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            bit_metrics[i] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm
            }
        
        return {
            "case_accuracy": case_accuracy,
            "bit_accuracy": bit_accuracy,
            "bit_metrics": bit_metrics,
            "total_cases": total_cases,
            "correct_cases": correct_cases
        }

    def generate_vhdl(self, filename: str = "mtncl_netlist.vhdl"):
        """Generate VHDL netlist for the trained network"""
        with open(filename, "w") as f:
            # Write VHDL libraries and packages
            f.write("library IEEE;\n")
            f.write("use IEEE.std_logic_1164.all;\n\n")
            
            # Write entity declaration
            f.write("entity mtncl_netlist is\n")
            f.write("  port (\n")
            # Generate input ports dynamically
            num_inputs = len(self.gates[0]) - 2  # Subtract constant gates
            for i in range(num_inputs):
                f.write(f"    input_{i} : in std_logic;\n")
            # Generate output ports dynamically
            for i, _ in enumerate(self.output_gates):
                f.write(f"    output_{i} : out std_logic;\n")
            f.write("    reset : in std_logic;  -- Active-low reset\n")
            f.write("    ki : in std_logic;     -- Input completion\n")
            f.write("    ko : out std_logic;    -- Output completion\n")
            f.write("    sleep : in std_logic   -- Sleep signal\n")
            f.write("  );\n")
            f.write("end entity mtncl_netlist;\n\n")
            
            # Write architecture
            f.write("architecture behavioral of mtncl_netlist is\n")
            
            # Write component declarations for each gate type used
            used_gate_types = {gate.gate_type for layer in self.gates[1:] for gate in layer 
                             if gate.gate_type not in {GateType.ALWAYS_ONE, GateType.ALWAYS_ZERO}}
            
            for gate_type in used_gate_types:
                f.write(f"  component {gate_type.name}\n")
                f.write("    port (\n")
                num_inputs = len([g for layer in self.gates[1:] for g in layer 
                                if g.gate_type == gate_type][0].inputs)
                f.write(f"      a : in std_logic_vector({num_inputs-1} downto 0);\n")
                f.write("      sleep : in std_logic;\n")
                f.write("      rst : in std_logic;\n")
                f.write("      ki : in std_logic;\n")
                f.write("      ko : out std_logic;\n")
                f.write("      z : out std_logic\n")
                f.write("    );\n")
                f.write("  end component;\n\n")
            
            # Write internal signals
            f.write("  -- Internal signals\n")
            for layer in self.gates[1:]:  # Skip input layer
                for gate in layer:
                    if gate.gate_type not in {GateType.ALWAYS_ONE, GateType.ALWAYS_ZERO}:
                        f.write(f"  signal {gate.name}_out : std_logic;\n")
                        f.write(f"  signal {gate.name}_ko : std_logic;\n")
            
            # Write constants
            f.write("  constant VDD : std_logic := '1';\n")
            f.write("  constant GND : std_logic := '0';\n")
            f.write("begin\n")
            
            # Generate gate instantiations
            for layer_idx, layer in enumerate(self.gates[1:], 1):  # Skip input layer
                for gate in layer:
                    if gate.gate_type in {GateType.ALWAYS_ONE, GateType.ALWAYS_ZERO}:
                        continue
                    
                    f.write(f"\n  -- {gate.name} ({gate.gate_type.name})\n")
                    f.write(f"  {gate.name}_inst: {gate.gate_type.name}\n")
                    f.write("    port map (\n")
                    
                    # Map inputs to a std_logic_vector
                    input_signals = []
                    for input_gate in gate.inputs:
                        if input_gate.gate_type == GateType.ALWAYS_ONE:
                            input_signals.append("VDD")
                        elif input_gate.gate_type == GateType.ALWAYS_ZERO:
                            input_signals.append("GND")
                        elif input_gate.layer == 0:  # Input gate
                            input_idx = self.gates[0].index(input_gate)
                            if input_idx < num_inputs:  # Only for actual inputs, not constants
                                input_signals.append(f"input_{input_idx}")
                        else:
                            input_signals.append(f"{input_gate.name}_out")
                    
                    f.write(f"      a => ({', '.join(input_signals)}),\n")
                    f.write("      sleep => sleep,\n")
                    f.write("      rst => reset,\n")
                    
                    # Handle completion signals
                    if gate in self.output_gates:
                        # Output gate
                        output_idx = self.output_gates.index(gate)
                        f.write("      ki => ki,\n")
                        f.write(f"      ko => ko,\n")  # Last gate drives main ko
                        f.write(f"      z => output_{output_idx}\n")
                    else:
                        # Find all gates in next layers that use this gate as input
                        next_gates = []
                        for next_layer in self.gates[layer_idx+1:]:
                            for next_gate in next_layer:
                                if gate in next_gate.inputs:
                                    next_gates.append(next_gate)
                        
                        if next_gates:
                            # Use the first gate's completion signal
                            f.write(f"      ki => {next_gates[0].name}_ko,\n")
                        else:
                            f.write("      ki => ki,\n")
                        f.write(f"      ko => {gate.name}_ko,\n")
                        f.write(f"      z => {gate.name}_out\n")
                    
                    f.write("    );\n")
            
            f.write("end architecture behavioral;\n")
