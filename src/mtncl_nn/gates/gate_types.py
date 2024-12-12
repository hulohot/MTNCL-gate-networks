"""
MTNCL Gate Types
This module defines all available MTNCL gate types and their behaviors.
"""

from enum import Enum, auto
from typing import List

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
            C = any(bool_inputs[2:])
            return float(A or (B and C))
        elif self == GateType.TH33w2:
            # AB + AC (A has weight 2)
            A, B = bool_inputs[:2]
            C = any(bool_inputs[2:])
            return float((A and B) or (A and C))
        
        # For other gates, implement a basic threshold check
        threshold = num_required // 2  # Default threshold is half of required inputs
        return float(sum(bool_inputs[:num_required]) >= threshold)

    @staticmethod
    def get_compatible_gates(num_inputs: int) -> set['GateType']:
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