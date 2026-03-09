#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum circuit examples
"""

import numpy as np


def main():
    # |0>
    zero = np.array([1, 0])

    # H, CX, CZ gate
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]).reshape(
        (2, 2, 2, 2)
    )
    CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]).reshape(
        (2, 2, 2, 2)
    )

    # initial state |000>
    state = np.kron(np.kron(zero, zero), zero).reshape((2, 2, 2))
    print(f"|Ψ> =\n{state}")
    print(f"     =\n{state.reshape((8,))}\n")

    # apply H_0
    state = np.einsum(H, [0, 3], state, [3, 1, 2])
    print(f"|Ψ> =\n{state}")
    print(f"    =\n{state.reshape((8,))}\n")

    # apply H_2
    state = np.einsum(H, [2, 3], state, [0, 1, 3])
    print(f"|Ψ> =\n{state}")
    print(f"    =\n{state.reshape((8,))}\n")

    # apply CX_01
    state = np.einsum(CX, [0, 1, 4, 5], state, [4, 5, 2])
    print(f"|Ψ> =\n{state}")
    print(f"    =\n{state.reshape((8,))}\n")

    # apply CZ_12
    state = np.einsum(CZ, [1, 2, 4, 5], state, [0, 4, 5])
    print(f"|Ψ> =\n{state}")
    print(f"    =\n{state.reshape((8,))}\n")

    # apply H_2
    state = np.einsum(H, [2, 3], state, [0, 1, 3])
    print(f"|Ψ> =\n{state}")
    print(f"    =\n{state.reshape((8,))}\n")


if __name__ == "__main__":
    main()
