import pytest

pytest.importorskip("cudaq")

from python.cudaq.shor import permutation
from python.cudaq.shor.permutation import (
    _controlled_single_bit_transposition,
    build_mod_exp_permutation,
)


class RecordingKernel:
    def __init__(self) -> None:
        self.calls = []

    def x(self, qubit) -> None:
        self.calls.append(("x", qubit))

    def cx(self, controls, target) -> None:
        self.calls.append(("cx", tuple(controls), target))


def test_controlled_single_bit_transposition() -> None:
    kernel = RecordingKernel()

    _controlled_single_bit_transposition(kernel, "c", ["q0", "q1", "q2"], 0b001, 0b101)

    assert kernel.calls == [
        ("x", "q1"),
        ("cx", ("c", "q0", "q1"), "q2"),
        ("x", "q1"),
    ]


def test_controlled_swap_permutation(monkeypatch) -> None:
    calls = []

    def fake_controlled_transposition(kernel, ctrl, target_qubits, a, b) -> None:
        calls.append((ctrl, tuple(target_qubits), a, b))

    monkeypatch.setattr(permutation, "controlled_transposition", fake_controlled_transposition)

    permutation.controlled_swap_permutation(
        object(),
        "c",
        ["q0", "q1", "q2"],
        {1: 2, 2: 1, 3: 4, 4: 3},
    )

    assert calls == [
        ("c", ("q0", "q1", "q2"), 1, 2),
        ("c", ("q0", "q1", "q2"), 3, 4),
    ]


def test_build_mod_exp_permutation() -> None:
    assert build_mod_exp_permutation(2, 5, 2) == {1: 4, 2: 3, 3: 2, 4: 1}
