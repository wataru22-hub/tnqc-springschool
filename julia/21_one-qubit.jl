#!/usr/bin/env julia
# One-qubit gate examples

function main()
    # |0>, |1>
    ket0 = reshape(ComplexF64[1, 0], 2, 1)
    ket1 = reshape(ComplexF64[0, 1], 2, 1)
    println("|0> =\n$(ket0)\n")
    println("|1> =\n$(ket1)\n")

    # X gate
    X = ComplexF64[0 1; 1 0]
    println("X =\n$(X)\n")
    println("X|0> = $(X * ket0)\n")
    println("X|1> = $(X * ket1)\n")

    # Y gate
    Y = ComplexF64[0 -1im; 1im 0]
    println("Y =\n$(Y)\n")
    println("Y|0> = $(Y * ket0)\n")
    println("Y|1> = $(Y * ket1)\n")

    # Z gate
    Z = ComplexF64[1 0; 0 -1]
    println("Z =\n$(Z)\n")
    println("Z|0> = $(Z * ket0)\n")
    println("Z|1> = $(Z * ket1)\n")

    # Hadamard gate
    H = ComplexF64[1 1; 1 -1] / sqrt(2)
    println("H =\n$(H)\n")
    println("H|0> = $(H * ket0)\n")
    println("H|1> = $(H * ket1)\n")
end

main()
