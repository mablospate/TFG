import math

import pytest

pytest.importorskip("cudaq")

from python.cudaq.shor.qft import apply_inverse_qft, apply_qft


class RecordingKernel:
    def __init__(self) -> None:
        self.calls = []

    def h(self, qubit) -> None:
        self.calls.append(("h", qubit))

    def cr1(self, angle, control, target) -> None:
        self.calls.append(("cr1", angle, control, target))

    def swap(self, left, right) -> None:
        self.calls.append(("swap", left, right))


def test_apply_qft() -> None:
    kernel = RecordingKernel()

    apply_qft(kernel, ["q0", "q1", "q2"], 3)

    assert kernel.calls == [
        ("h", "q0"),
        ("cr1", math.pi / 2, "q1", "q0"),
        ("cr1", math.pi / 4, "q2", "q0"),
        ("h", "q1"),
        ("cr1", math.pi / 2, "q2", "q1"),
        ("h", "q2"),
        ("swap", "q0", "q2"),
    ]


def test_apply_inverse_qft() -> None:
    kernel = RecordingKernel()

    apply_inverse_qft(kernel, ["q0", "q1", "q2"], 3)

    assert kernel.calls == [
        ("swap", "q0", "q2"),
        ("h", "q2"),
        ("cr1", -math.pi / 2, "q2", "q1"),
        ("h", "q1"),
        ("cr1", -math.pi / 4, "q2", "q0"),
        ("cr1", -math.pi / 2, "q1", "q0"),
        ("h", "q0"),
    ]
