import pytest

pytest.importorskip("cirq")

from python.cirq.shor.modular_exp import ModularExp


def test_modular_exp_apply() -> None:
    gate = ModularExp(target_size=3, exponent_size=2, base=2, modulus=5)

    assert gate.apply(3, 1) == (1, 1)
    assert gate.apply(4, 2) == (1, 2)
    assert gate.apply(5, 3) == (5, 3)


def test_modular_exp_metadata() -> None:
    gate = ModularExp(target_size=3, exponent_size=2, base=2, modulus=5)

    assert gate.registers() == [[2, 2, 2], [2, 2]]
    assert gate.with_registers([0, 1, 2], [0, 1]) == gate
    assert repr(gate) == "ModularExp(base=2, modulus=5)"
    assert hash(gate) == hash(ModularExp(3, 2, 2, 5))
