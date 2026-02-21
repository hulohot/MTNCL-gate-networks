"""
MTNCL Neural Network Framework
A framework for building neural networks using Multi-Threshold Null Convention Logic (MTNCL) gates.
"""

from .core.network import MTNCLNetwork
from .gates.gate import MTNCLGate
from .gates.gate_types import GateType
from .ensemble import MTNCLEnsemble
from .multistart import train_multistart, MultiStartResult
from .preprocessing import build_features

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    'MTNCLNetwork',
    'MTNCLGate',
    'GateType',
    'MTNCLEnsemble',
    'train_multistart',
    'MultiStartResult',
    'build_features',
] 