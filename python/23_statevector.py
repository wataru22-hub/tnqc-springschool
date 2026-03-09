#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum circuit examples
"""

import numpy as np


def main():
    # |0>
    zero = np.array([1, 0])

    # Id, H, CX, CZ gate
    Id = np.eye(2)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    # initial state |000>
    state = np.kron(np.kron(zero, zero), zero)
    print(f"|Ψ> =\n{state}\n")

    # apply H_0
    Op = np.kron(np.kron(H, Id), Id)
    state = Op @ state
    print(f"Op =\n{Op}\n")
    print(f"|Ψ> =\n{state}\n")

    # apply H_2
    Op = np.kron(np.kron(Id, Id), H)
    state = Op @ state
    print(f"Op =\n{Op}\n")
    print(f"|Ψ> =\n{state}\n")

    # apply CX_01
    Op = np.kron(CX, Id)
    state = Op @ state
    print(f"Op =\n{Op}\n")
    print(f"|Ψ> =\n{state}\n")

    # apply CZ_12
    """CZ gate"""
    Op = np.kron(Id, CZ)
    state = Op @ state
    print(f"Op =\n{Op}\n")
    print(f"|Ψ> =\n{state}\n")

    # apply H_2
    Op = np.kron(np.kron(Id, Id), H)
    state = Op @ state
    print(f"Op =\n{Op}\n")
    print(f"|Ψ> =\n{state}\n")


if __name__ == "__main__":
    main()
