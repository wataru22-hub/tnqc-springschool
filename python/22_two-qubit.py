#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-qubit gate examples
"""

import numpy as np


def main():
    # |00>, |01>, |10>, |11>
    ket00 = np.array([[1], [0], [0], [0]], dtype=complex)
    ket01 = np.array([[0], [1], [0], [0]], dtype=complex)
    ket10 = np.array([[0], [0], [1], [0]], dtype=complex)
    ket11 = np.array([[0], [0], [0], [1]], dtype=complex)
    print(f"|00> =\n{ket00}\n")
    print(f"|01> =\n{ket01}\n")
    print(f"|10> =\n{ket10}\n")
    print(f"|11> =\n{ket11}\n")

    """CNOT_01 gate"""
    CNOT_01 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    print(f"CNOT_01 =\n{CNOT_01}\n")
    print(f"CNOT_01|00> = {CNOT_01 @ ket00}\n")
    print(f"CNOT_01|01> = {CNOT_01 @ ket01}\n")
    print(f"CNOT_01|10> = {CNOT_01 @ ket10}\n")
    print(f"CNOT_01|11> = {CNOT_01 @ ket11}\n")

    """CNOT_12 gate"""
    CNOT_12 = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
    )
    print(f"CNOT_12 =\n{CNOT_12}\n")
    print(f"CNOT_12|00> = {CNOT_12 @ ket00}\n")
    print(f"CNOT_12|01> = {CNOT_12 @ ket01}\n")
    print(f"CNOT_12|10> = {CNOT_12 @ ket10}\n")
    print(f"CNOT_12|11> = {CNOT_12 @ ket11}\n")

    """CZ gate"""
    CZ = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
    )
    print(f"CZ =\n{CZ}\n")
    print(f"CZ|00> = {CZ @ ket00}\n")
    print(f"CZ|01> = {CZ @ ket01}\n")
    print(f"CZ|10> = {CZ @ ket10}\n")
    print(f"CZ|11> = {CZ @ ket11}\n")

    """Swap gate"""
    Swap = CNOT_01 @ CNOT_12 @ CNOT_01
    print(f"Swap =\n{Swap}\n")
    print(f"Swap|00> = {Swap @ ket00}\n")
    print(f"Swap|01> = {Swap @ ket01}\n")
    print(f"Swap|10> = {Swap @ ket10}\n")
    print(f"Swap|11> = {Swap @ ket11}\n")


if __name__ == "__main__":
    main()
