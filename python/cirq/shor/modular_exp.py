import cirq


class ModularExp(cirq.ArithmeticGate):
    """
    Modular exponentiation gate that computes |x>|y> -> |x>|y * base^x mod modulus>.

    This is the key quantum operation for Shor's algorithm: it performs controlled
    modular exponentiation using Cirq's ArithmeticGate abstraction, which defines
    the mathematical semantics and lets the simulator compute the result classically.

    Args:
        target_size: Number of qubits in the target register.
        exponent_size: Number of qubits in the exponent register.
        base: The base for modular exponentiation (A in A^x mod N).
        modulus: The modulus (N in A^x mod N).
    """

    def __init__(
        self,
        target_size: int,
        exponent_size: int,
        base: int,
        modulus: int,
    ) -> None:
        self.target_size = target_size
        self.exponent_size = exponent_size
        self.base = base
        self.modulus = modulus

    def registers(self) -> list[int | list[int]]:
        return [[2] * self.target_size, [2] * self.exponent_size]

    def with_registers(self, *new_registers) -> "ModularExp":
        if isinstance(new_registers[0], int):
            target_size = 1
        else:
            target_size = len(new_registers[0])
        if isinstance(new_registers[1], int):
            exponent_size = 1
        else:
            exponent_size = len(new_registers[1])
        return ModularExp(
            target_size=target_size,
            exponent_size=exponent_size,
            base=self.base,
            modulus=self.modulus,
        )

    def apply(self, target_value: int, exponent_value: int) -> tuple[int, int]:
        if target_value < self.modulus:
            return (
                (target_value * pow(self.base, exponent_value, self.modulus))
                % self.modulus,
                exponent_value,
            )
        return target_value, exponent_value

    def __repr__(self) -> str:
        return f"ModularExp(base={self.base}, modulus={self.modulus})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModularExp):
            return NotImplemented
        return (
            self.target_size == other.target_size
            and self.exponent_size == other.exponent_size
            and self.base == other.base
            and self.modulus == other.modulus
        )

    def __hash__(self) -> int:
        return hash((self.target_size, self.exponent_size, self.base, self.modulus))
