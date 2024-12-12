"""
MTNCL Neural Network Implementation
This module provides the core neural network implementation using MTNCL gates.
"""

from typing import List, Dict, Tuple, Set
import random
from ..gates.gate import MTNCLGate
from ..gates.gate_types import GateType

class MTNCLNetwork:
    """Neural network implementation using MTNCL gates"""
    
    def __init__(self, num_inputs: int, num_outputs: int, hidden_layers: List[int], verbose: bool = False):
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
        self.verbose = verbose
        
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
        
        if self.verbose:
            print(f"Created network with {len(self.gates)} layers:")
            for i, layer in enumerate(self.gates):
                print(f"  Layer {i}: {len(layer)} gates")
    
    def _create_gate_name(self, prefix: str = "gate") -> str:
        """Create a unique gate name"""
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
    
    def train(self, X: List[List[float]], y: List[List[float]], iterations: int = 1000,
             temperature_start: float = 0.8, temperature_end: float = 0.1) -> None:
        """Train the network using a modified simulated annealing approach"""
        best_error = float('inf')
        best_accuracy = 0.0
        best_config = None
        plateau_counter = 0
        max_plateau = 50  # Number of iterations without improvement before resetting
        
        if self.verbose:
            print("\nStarting training...")
            print(f"Parameters: iterations={iterations}, temp_start={temperature_start}, temp_end={temperature_end}")
        
        # Initialize with smart gate selection
        self._smart_gate_init(temperature=1.0)
        error, accuracy = self.evaluate(X, y)
        
        for iteration in range(iterations):
            # Anneal temperature
            temperature = temperature_start - (temperature_start - temperature_end) * (iteration / iterations)
            
            # Store current configuration
            current_config = [(gate.gate_type, gate.inputs[:]) for layer in self.gates[1:] for gate in layer]
            
            # Perform multiple local modifications
            num_modifications = max(1, int(4 * temperature))
            modified_gates = set()
            
            for _ in range(num_modifications):
                layer_idx = random.randint(1, len(self.gates) - 1)
                gate = random.choice(self.gates[layer_idx])
                
                if gate not in modified_gates:
                    modified_gates.add(gate)
                    
                    if random.random() < 0.7:  # 70% chance to change gate type
                        old_type = gate.gate_type
                        compatible_gates = self.get_gate_options(gate)
                        new_type = random.choice([g for g in compatible_gates if g != old_type])
                        gate.gate_type = new_type
                    else:  # 30% chance to modify inputs
                        available_inputs = [g for l in self.gates[:layer_idx] for g in l]
                        if len(available_inputs) > 2:
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
                    if self.verbose:
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
                if self.verbose:
                    print(f"\nReset at iteration {iteration} due to plateau")
                self._smart_gate_init(temperature=1.0)
                plateau_counter = 0
            
            # Store training history
            self.training_history['errors'].append(new_error)
            self.training_history['accuracies'].append(new_accuracy)
            
            if iteration % 100 == 0:
                if self.verbose:
                    print(f"Iteration {iteration}: Error = {new_error:.4f}, Accuracy = {new_accuracy*100:.1f}%, "
                          f"Best Accuracy = {best_accuracy*100:.1f}%")
                else:
                    print(f"Training progress: {iteration/iterations*100:.1f}% (Accuracy: {new_accuracy*100:.1f}%)", end='\r')
        
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
    
    def _print_debug_info(self, iteration: int, error: float, accuracy: float, X: List[List[float]], y: List[List[float]]):
        """Print detailed debug information"""
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}")
        print(f"Error: {error:.4f}, Accuracy: {accuracy*100:.2f}%")
        
        # Print layer-by-layer analysis
        print("\nLayer Analysis:")
        layer_stats = []
        for layer_idx, layer in enumerate(self.gates[1:], 1):
            print(f"\nLayer {layer_idx}:")
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
                
                print(f"  {gate.name} ({gate.gate_type.name}):")
                print(f"    Avg Output: {avg_output:.2f}")
                print(f"    {'[STUCK HIGH]' if always_high else '[STUCK LOW]' if always_low else '[ACTIVE]'}")
                print(f"    Switching Rate: {switching/(len(outputs)-1)*100:.1f}%")
                print(f"    Inputs: {[g.name for g in gate.inputs]}")
            
            layer_stats.append(layer_stat)
        
        self.training_history['layer_stats'].append({
            'iteration': iteration,
            'error': error,
            'accuracy': accuracy,
            'layers': layer_stats
        }) 