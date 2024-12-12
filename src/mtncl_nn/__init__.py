"""
MTNCL Neural Network Framework
A framework for building neural networks using Multi-Threshold Null Convention Logic (MTNCL) gates.
"""

from .core.network import MTNCLNetwork
from .gates.gate import MTNCLGate
from .gates.gate_types import GateType

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = ['MTNCLNetwork', 'MTNCLGate', 'GateType'] 