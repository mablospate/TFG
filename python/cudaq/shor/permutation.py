import cudaq


def _controlled_single_bit_transposition(
    kernel: cudaq.Kernel,
    ctrl,
    target_qubits: list,
    a: int,
    b: int,
) -> None:
    """
    Implement a controlled transposition of |a> and |b> where a and b differ in
    exactly one bit.

    Args:
        kernel: CUDA-Q kernel being built.
        ctrl: External control qubit.
        target_qubits: List of qubit references.
        a: First state (integer).
        b: Second state (integer).
    """
    n = len(target_qubits)
    diff = a ^ b
    assert diff != 0 and (diff & (diff - 1)) == 0, "a and b must differ in exactly one bit"
    flip_bit = diff.bit_length() - 1

    other_positions = [i for i in range(n) if i != flip_bit]

    for pos in other_positions:
        if not ((a >> pos) & 1):
            kernel.x(target_qubits[pos])

    controls = [ctrl] + [target_qubits[pos] for pos in other_positions]
    kernel.cx(controls, target_qubits[flip_bit])

    for pos in other_positions:
        if not ((a >> pos) & 1):
            kernel.x(target_qubits[pos])


def controlled_transposition(
    kernel: cudaq.Kernel,
    ctrl,
    target_qubits: list,
    a: int,
    b: int,
) -> None:
    """
    Implement a controlled transposition |a> <-> |b> on the target register.
    Uses recursive decomposition into single-bit transpositions.

    Args:
        kernel: CUDA-Q kernel being built.
        ctrl: Control qubit reference.
        target_qubits: List of qubit references for the target register.
        a: First basis state (integer).
        b: Second basis state (integer).
    """
    diff_bits = a ^ b
    if diff_bits == 0:
        return

    n = len(target_qubits)
    diff_positions = [i for i in range(n) if (diff_bits >> i) & 1]

    if len(diff_positions) == 1:
        _controlled_single_bit_transposition(kernel, ctrl, target_qubits, a, b)
    else:
        pivot = diff_positions[0]
        a_prime = a ^ (1 << pivot)
        controlled_transposition(kernel, ctrl, target_qubits, a, a_prime)
        controlled_transposition(kernel, ctrl, target_qubits, a_prime, b)
        controlled_transposition(kernel, ctrl, target_qubits, a, a_prime)


def controlled_swap_permutation(
    kernel: cudaq.Kernel,
    ctrl,
    target_qubits: list,
    permutation: dict[int, int],
) -> None:
    """
    Implement a controlled permutation of basis states on the target register.
    Decomposed into disjoint cycles, then into transpositions.

    Args:
        kernel: CUDA-Q kernel being built.
        ctrl: Control qubit reference.
        target_qubits: List of qubit references for the target register.
        permutation: Mapping from input state (int) to output state (int).
    """
    visited = set()

    for start in sorted(permutation.keys()):
        if start in visited:
            continue
        cycle = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = permutation.get(current, current)

        if len(cycle) <= 1:
            continue

        for idx in range(1, len(cycle)):
            controlled_transposition(
                kernel, ctrl, target_qubits, cycle[0], cycle[idx]
            )


def build_mod_exp_permutation(A: int, N: int, power: int) -> dict[int, int]:
    """
    Build the permutation that maps |y> -> |A^power * y mod N> for 0 <= y < N.

    Args:
        A: Base for modular exponentiation.
        N: Modulus.
        power: Exponent.
    Returns:
        dict mapping input state to output state.
    """
    a_power = pow(A, power, N)
    permutation = {}
    for y in range(N):
        target = (a_power * y) % N
        if y != target:
            permutation[y] = target
    return permutation
