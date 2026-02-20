"""
MTNCL Neural Network Implementation
This module provides the core neural network implementation using MTNCL gates.
"""

from typing import List, Dict, Tuple, Set, Any
import json
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
            MTNCLGate(name=self._create_gate_name("const"), layer=0, inputs=[], gate_type=GateType.ALWAYS_ONE, value=1.0),
            MTNCLGate(name=self._create_gate_name("const"), layer=0, inputs=[], gate_type=GateType.ALWAYS_ZERO, value=0.0)
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
                    compatible_preferred = [g for g in preferred_gates['arithmetic'] if g in self.get_gate_options(gate)]
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
        _error, _accuracy = self.evaluate(X, y)

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
                        new_choices = [g for g in compatible_gates if g != old_type]
                        if new_choices:
                            gate.gate_type = random.choice(new_choices)
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
                    best_config = [(gate.gate_type, gate.inputs[:]) for layer in self.gates[1:] for gate in layer]
                    plateau_counter = 0
                    if self.verbose:
                        self._print_debug_info(iteration, new_error, new_accuracy, X, y)
            else:
                # Revert changes
                for gate, (old_type, old_inputs) in zip((g for l in self.gates[1:] for g in l), current_config):
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

        # Restore best-seen configuration before returning
        if best_config is not None:
            for gate, (best_type, best_inputs) in zip((g for l in self.gates[1:] for g in l), best_config):
                gate.gate_type = best_type
                gate.inputs = best_inputs

        print(f"Training completed. Best accuracy: {best_accuracy*100:.1f}%")

    def evaluate(self, X: List[List[float]], y: List[List[float]]) -> Tuple[float, float]:
        """Evaluate network performance on given inputs"""
        total_error = 0.0
        correct_outputs = 0

        for x_i, y_i in zip(X, y):
            outputs = self.forward(x_i)
            if outputs == y_i:
                correct_outputs += 1
            total_error += 0 if outputs == y_i else 1

        error_rate = total_error / len(X)
        accuracy = correct_outputs / len(X)

        return error_rate, accuracy

    def forward(self, x: List[float]) -> List[float]:
        """Forward propagate input through the network"""
        # Set input values (excluding two const gates at end of layer 0)
        for input_gate, value in zip(self.gates[0][:-2], x):
            input_gate.value = value

        # Ensure constants always hold deterministic values
        self.gates[0][-2].value = 1.0
        self.gates[0][-1].value = 0.0

        raw_outputs = [gate.forward() for gate in self.output_gates]
        return self._to_one_hot(raw_outputs)

    def _to_one_hot(self, outputs: List[float]) -> List[float]:
        """Convert raw outputs to one-hot encoding (binary outputs)."""
        if not outputs:
            return []

        max_val = max(outputs)
        max_idx = outputs.index(max_val)
        return [1.0 if i == max_idx else 0.0 for i in range(len(outputs))]

    def predict(self, X: List[List[float]]) -> List[List[float]]:
        """Predict one-hot outputs for a batch of inputs."""
        return [self.forward(row) for row in X]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model structure to a JSON-compatible dict."""
        gate_to_name = {g: g.name for layer in self.gates for g in layer}
        return {
            "num_layers": len(self.gates),
            "layers": [
                [
                    {
                        "name": gate.name,
                        "layer": gate.layer,
                        "gate_type": gate.gate_type.name if gate.gate_type else None,
                        "inputs": [gate_to_name[inp] for inp in gate.inputs],
                        "value": gate.value,
                    }
                    for gate in layer
                ]
                for layer in self.gates
            ],
            "training_history": self.training_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], verbose: bool = False) -> "MTNCLNetwork":
        """Reconstruct a model from a dict produced by to_dict."""
        num_inputs = len(data["layers"][0]) - 2
        num_outputs = len(data["layers"][-1])
        hidden_layers = [len(layer) for layer in data["layers"][1:-1]]

        model = cls(num_inputs=num_inputs, num_outputs=num_outputs, hidden_layers=hidden_layers, verbose=verbose)

        # Map runtime gate objects by name
        runtime_map = {g.name: g for layer in model.gates for g in layer}
        source_map = {g["name"]: g for layer in data["layers"] for g in layer}

        # Set gate attributes from serialized data
        for name, gate_data in source_map.items():
            if name not in runtime_map:
                continue
            gate_obj = runtime_map[name]
            gate_obj.gate_type = GateType[gate_data["gate_type"]] if gate_data["gate_type"] else None
            gate_obj.value = gate_data.get("value", 0.0)

        # Rewire inputs after all objects exist
        for name, gate_data in source_map.items():
            if name not in runtime_map:
                continue
            gate_obj = runtime_map[name]
            gate_obj.inputs = [runtime_map[i_name] for i_name in gate_data.get("inputs", []) if i_name in runtime_map]

        model.training_history = data.get("training_history", model.training_history)
        return model

    def save(self, path: str) -> None:
        """Save model to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, verbose: bool = False) -> "MTNCLNetwork":
        """Load model from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, verbose=verbose)

    def _gate_expr(self, gate: MTNCLGate) -> str:
        """Return a Verilog boolean expression for a gate output."""
        if gate.gate_type is None:
            return "1'b0"

        gt = gate.gate_type
        names = [inp.name for inp in gate.inputs]

        if gt == GateType.ALWAYS_ONE:
            return "1'b1"
        if gt == GateType.ALWAYS_ZERO:
            return "1'b0"

        # Aliases for readability in emitted expressions
        A = names[0] if len(names) > 0 else "1'b0"
        B = names[1] if len(names) > 1 else "1'b0"

        if gt == GateType.TH12:
            return f"({A} | {B})"
        if gt == GateType.TH22:
            return f"({A} & {B})"
        if gt == GateType.TH13:
            c3 = " | ".join(names[:3]) if names else "1'b0"
            return f"({c3})"
        if gt == GateType.TH23:
            c3 = names[:3]
            while len(c3) < 3:
                c3.append("1'b0")
            a, b, c = c3
            return f"(({a} & {b}) | ({a} & {c}) | ({b} & {c}))"
        if gt == GateType.TH33:
            c3 = names[:3]
            while len(c3) < 3:
                c3.append("1'b0")
            a, b, c = c3
            return f"({a} & {b} & {c})"
        if gt == GateType.THxor0:
            return f"({A} ^ {B})"
        if gt == GateType.THand0:
            return f"({A} & {B})"
        if gt == GateType.TH23w2:
            c = " | ".join(names[2:]) if len(names) > 2 else "1'b0"
            return f"({A} | ({B} & ({c})))"
        if gt == GateType.TH33w2:
            c = " | ".join(names[2:]) if len(names) > 2 else "1'b0"
            return f"(({A} & {B}) | ({A} & ({c})))"

        # Fallback: match compute_output default threshold behavior.
        # num_required parsed from gate enum name THxy... (x = input count), threshold = x//2.
        try:
            num_required = int(gt.name[2]) if len(gt.name) >= 3 else 2
        except ValueError:
            num_required = 2

        threshold = max(1, num_required // 2)
        terms = names[:num_required]
        while len(terms) < num_required:
            terms.append("1'b0")

        # Build sum of 1-bit terms and compare against threshold.
        summed = " + ".join([f"({t} ? 1 : 0)" for t in terms]) if terms else "0"
        return f"(({summed}) >= {threshold})"

    def to_verilog(self, module_name: str = "mtncl_net") -> str:
        """Emit a synthesizable-ish combinational Verilog netlist for the current network."""
        input_gates = self.gates[0][:-2]  # exclude constants
        output_gates = self.output_gates
        internal_gates = [g for layer in self.gates[1:] for g in layer if g not in output_gates]

        lines: List[str] = []
        lines.append(f"module {module_name}(")
        lines.append("  input  wire [" + str(len(input_gates) - 1) + ":0] in_bits,")
        lines.append("  output wire [" + str(len(output_gates) - 1) + ":0] out_bits")
        lines.append(");")
        lines.append("")

        # Input aliases
        for i, g in enumerate(input_gates):
            lines.append(f"  wire {g.name} = in_bits[{i}];")

        # Constants
        lines.append(f"  wire {self.gates[0][-2].name} = 1'b1;")
        lines.append(f"  wire {self.gates[0][-1].name} = 1'b0;")
        lines.append("")

        # Internal + output logic
        for g in internal_gates + output_gates:
            expr = self._gate_expr(g)
            lines.append(f"  wire {g.name} = {expr};")

        lines.append("")
        for i, g in enumerate(output_gates):
            lines.append(f"  assign out_bits[{i}] = {g.name};")

        lines.append("endmodule")
        lines.append("")
        return "\n".join(lines)

    def to_dot(self, graph_name: str = "mtncl_net") -> str:
        """Emit a Graphviz DOT representation of the network netlist."""
        lines: List[str] = []
        lines.append(f"digraph {graph_name} {{")
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=rounded];")

        for layer_idx, layer in enumerate(self.gates):
            lines.append(f"  subgraph cluster_{layer_idx} {{")
            lines.append(f"    label=\"Layer {layer_idx}\";")
            lines.append("    color=lightgrey;")
            for gate in layer:
                gt = gate.gate_type.name if gate.gate_type else ("INPUT" if layer_idx == 0 else "UNSET")
                lines.append(f"    \"{gate.name}\" [label=\"{gate.name}\\n{gt}\"];")
            lines.append("  }")

        for layer in self.gates[1:]:
            for gate in layer:
                for inp in gate.inputs:
                    lines.append(f"  \"{inp.name}\" -> \"{gate.name}\";")

        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _print_debug_info(self, iteration: int, error: float, accuracy: float, X: List[List[float]], y: List[List[float]]):
        """Print detailed debug information"""
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}")
        print(f"Error: {error:.4f}, Accuracy: {accuracy*100:.2f}%")

        layer_stats = []
        for layer_idx, layer in enumerate(self.gates[1:], 1):
            layer_stat = {'layer': layer_idx, 'gates': []}
            for gate in layer:
                outputs = []
                for x in X:
                    self.forward(x)
                    outputs.append(gate.value)

                avg_output = sum(outputs) / len(outputs)
                always_high = all(o > 0.5 for o in outputs)
                always_low = all(o < 0.5 for o in outputs)
                switching = sum(1 for i in range(1, len(outputs)) if outputs[i] != outputs[i-1])

                gate_stat = {
                    'name': gate.name,
                    'type': gate.gate_type.name if gate.gate_type else "UNSET",
                    'avg_output': avg_output,
                    'stuck': 'HIGH' if always_high else 'LOW' if always_low else 'NO',
                    'switching_rate': switching / max(1, (len(outputs) - 1)),
                }
                layer_stat['gates'].append(gate_stat)

            layer_stats.append(layer_stat)

        self.training_history['layer_stats'].append({
            'iteration': iteration,
            'error': error,
            'accuracy': accuracy,
            'layers': layer_stats,
        })
