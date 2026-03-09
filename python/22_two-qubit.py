#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-qubit gate examples
"""

import numpy as np


def main():
    # |00>, |01>, |10>, |11>
    ket00 = np.array([[1], [0], [0], [0]])
    ket01 = np.array([[0], [1], [0], [0]])
    ket10 = np.array([[0], [0], [1], [0]])
    ket11 = np.array([[0], [0], [0], [1]])
    print(f"|00> =\n{ket00}\n")
    print(f"|01> =\n{ket01}\n")
    print(f"|10> =\n{ket10}\n")
    print(f"|11> =\n{ket11}\n")

    """CX gate"""
    CX12 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    print(f"CX12 =\n{CX12}\n")
    print(f"CX12|00> = {CX12 @ ket00}\n")
    print(f"CX12|01> = {CX12 @ ket01}\n")
    print(f"CX12|10> = {CX12 @ ket10}\n")
    print(f"CX12|11> = {CX12 @ ket11}\n")

    """CX21 gate"""
    CX21 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    print(f"CX21 =\n{CX21}\n")
    print(f"CX21|00> = {CX21 @ ket00}\n")
    print(f"CX21|01> = {CX21 @ ket01}\n")
    print(f"CX21|10> = {CX21 @ ket10}\n")
    print(f"CX21|11> = {CX21 @ ket11}\n")

    """CZ gate"""
    CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    print(f"CZ =\n{CZ}\n")
    print(f"CZ|00> = {CZ @ ket00}\n")
    print(f"CZ|01> = {CZ @ ket01}\n")
    print(f"CZ|10> = {CZ @ ket10}\n")
    print(f"CZ|11> = {CZ @ ket11}\n")

    """Swap gate"""
    Swap = CX12 @ CX21 @ CX12
    print(f"Swap =\n{Swap}\n")
    print(f"Swap|00> = {Swap @ ket00}\n")
    print(f"Swap|01> = {Swap @ ket01}\n")
    print(f"Swap|10> = {Swap @ ket10}\n")
    print(f"Swap|11> = {Swap @ ket11}\n")


if __name__ == "__main__":
    main()
