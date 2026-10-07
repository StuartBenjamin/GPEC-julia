using Test
using LinearAlgebra

# The Galerkin banded solve must survive the ~25-decade diagonal spread of the assembled system:
# A = D·A0·D with A0 well conditioned, weakly coupled, and D alternating 4e-8 / 1e5 drives
# unscaled partial-pivoting LU to ~1e-4 relative error; Jacobi scaling recovers ~1e-15.
@testset "Galerkin banded solve" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

    n, kl = 24, 2
    A0 = zeros(ComplexF64, n, n)
    for j in 1:n, i in max(1, j - kl):min(n, j + kl)
        A0[i, j] = i == j ? 2 + 0.1 * sin(j) : 1e-5 * cis(0.3i + 0.7j)
    end
    A0 = (A0 + A0') / 2
    d = [isodd(j) ? 4e-8 : 1e5 for j in 1:n]
    A = Diagonal(d) * A0 * Diagonal(d)
    x = ComplexF64[cis(0.1j) for j in 1:n] ./ d
    b = A * hcat(x, 2x)

    # LAPACK band storage as GalWorkspace lays it out per solver
    function band(A, solver)
        lu = solver == "LU"
        m = size(A, 2)
        off = lu ? 2kl + 1 : 1
        ab = zeros(ComplexF64, lu ? 3kl + 1 : kl + 1, m)
        for j in 1:m, i in max(1, j - (lu ? kl : 0)):min(m, j + kl)
            ab[off+i-j, j] = A[i, j]
        end
        return ab
    end

    for solver in ("LU", "cholesky")
        @testset "$solver" begin
            sol = copy(b)
            FFS.gal_banded_solve!(band(A, solver), sol, kl, kl, solver)
            @test maximum(abs.(sol[:, 1] .- x) ./ abs.(x)) < 1e-12
            @test sol[:, 2] ≈ 2 * sol[:, 1]  # every right-hand side is un-scaled
        end
    end

    @testset "zero diagonal entry" begin
        Z = ComplexF64[0 1 0; 1 1 1; 0 1 3]
        z = ComplexF64[1, -2, 0.5]
        sol = Z * z
        FFS.gal_banded_solve!(band(Z, "LU"), sol, kl, kl, "LU")
        @test sol ≈ z
    end
end
