#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-qubit gate examples
"""

import numpy as np


def main():
    # |0>, |1>
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)
    print(f"|0> =\n{ket0}\n")
    print(f"|1> =\n{ket1}\n")

    """X gate"""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    print(f"X =\n{X}\n")
    print(f"X|0> = {X @ ket0}\n")
    print(f"X|1> = {X @ ket1}\n")

    """Y gate"""
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    print(f"Y =\n{Y}\n")
    print(f"Y|0> = {Y @ ket0}\n")
    print(f"Y|1> = {Y @ ket1}\n")

    """Z gate"""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    print(f"Z =\n{Z}\n")
    print(f"Z|0> = {Z @ ket0}\n")
    print(f"Z|1> = {Z @ ket1}\n")

    """Hadamard gate"""
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    print(f"H =\n{H}\n")
    print(f"H|0> = {H @ ket0}\n")
    print(f"H|1> = {H @ ket1}\n")


if __name__ == "__main__":
    main()
