#!/usr/bin/env julia
# Two-qubit gate examples

function main()
    # |00>, |01>, |10>, |11>
    ket00 = reshape(ComplexF64[1, 0, 0, 0], 4, 1)
    ket01 = reshape(ComplexF64[0, 1, 0, 0], 4, 1)
    ket10 = reshape(ComplexF64[0, 0, 1, 0], 4, 1)
    ket11 = reshape(ComplexF64[0, 0, 0, 1], 4, 1)
    println("|00> =\n$(ket00)\n")
    println("|01> =\n$(ket01)\n")
    println("|10> =\n$(ket10)\n")
    println("|11> =\n$(ket11)\n")

    # CX12 gate
    CX12 = ComplexF64[
        1 0 0 0
        0 1 0 0
        0 0 0 1
        0 0 1 0
    ]
    println("CX12 =\n$(CX12)\n")
    println("CX12|00> = $(CX12 * ket00)\n")
    println("CX12|01> = $(CX12 * ket01)\n")
    println("CX12|10> = $(CX12 * ket10)\n")
    println("CX12|11> = $(CX12 * ket11)\n")

    # CX21 gate
    CX21 = ComplexF64[
        1 0 0 0
        0 0 0 1
        0 0 1 0
        0 1 0 0
    ]
    println("CX21 =\n$(CX21)\n")
    println("CX21|00> = $(CX21 * ket00)\n")
    println("CX21|01> = $(CX21 * ket01)\n")
    println("CX21|10> = $(CX21 * ket10)\n")
    println("CX21|11> = $(CX21 * ket11)\n")

    # CZ gate
    CZ = ComplexF64[
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 -1
    ]
    println("CZ =\n$(CZ)\n")
    println("CZ|00> = $(CZ * ket00)\n")
    println("CZ|01> = $(CZ * ket01)\n")
    println("CZ|10> = $(CZ * ket10)\n")
    println("CZ|11> = $(CZ * ket11)\n")

    # Swap gate
    Swap = CX12 * CX21 * CX12
    println("Swap =\n$(Swap)\n")
    println("Swap|00> = $(Swap * ket00)\n")
    println("Swap|01> = $(Swap * ket01)\n")
    println("Swap|10> = $(Swap * ket10)\n")
    println("Swap|11> = $(Swap * ket11)\n")
end

main()
