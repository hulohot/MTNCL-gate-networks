"""
MTNCL Gate Implementation
This module provides the core gate implementation for MTNCL neural networks.
"""

from dataclasses import dataclass
from typing import List, Optional
from .gate_types import GateType

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