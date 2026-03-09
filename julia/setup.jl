using Pkg
Pkg.activate(".")
Pkg.add([
    "ImageIO",
    "Images",
    "LinearAlgebra",
    "Plots",
    "Printf",
    "Random",
    "TensorOperations"
])
Pkg.instantiate()
Pkg.precompile()
