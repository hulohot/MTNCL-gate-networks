from mtncl_nn.gates.gate_types import GateType


def test_basic_gate_outputs():
    assert GateType.TH12.compute_output([0, 0]) == 0.0
    assert GateType.TH12.compute_output([1, 0]) == 1.0
    assert GateType.TH22.compute_output([1, 1]) == 1.0
    assert GateType.TH22.compute_output([1, 0]) == 0.0


def test_xor_and_constants():
    assert GateType.THxor0.compute_output([0, 1]) == 1.0
    assert GateType.THxor0.compute_output([1, 1]) == 0.0
    assert GateType.ALWAYS_ONE.compute_output([]) == 1.0
    assert GateType.ALWAYS_ZERO.compute_output([]) == 0.0


def test_compatible_gates_grow_with_inputs():
    g2 = GateType.get_compatible_gates(2)
    g4 = GateType.get_compatible_gates(4)
    assert GateType.TH12 in g2
    assert GateType.TH24 in g4
    assert len(g4) >= len(g2)
