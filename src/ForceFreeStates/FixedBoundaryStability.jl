"""
    evaluate_stability_criterion!(odet, profiles; first_step=1) -> nzero

Fill `odet.crit_store` over all saved steps and return the number of zero crossings of the
critical eigenvalue (each signals a fixed-boundary instability) after step `first_step`. Warns with
a count if W⁻¹ was non-Hermitian beyond tolerance at any step.
"""
function evaluate_stability_criterion!(odet::OdeState, profiles::Equilibrium.ProfileSplines; first_step::Int=1)

    resize!(odet.crit_store, odet.step)
    nzero = 0
    nonherm_count = 0

    for istep in 1:odet.step
        ncross, nonherm = check_for_zero_crossings!(odet, profiles, istep; first_step)
        nzero += ncross
        if nonherm
            nonherm_count += 1
        end
    end

    if nonherm_count > 0
        @warn "W inverse matrix was non-Hermitian beyond tolerance at $nonherm_count integration step(s)"
    end

    return nzero
end

"""
    check_for_zero_crossings!(odet, profiles, istep; first_step=1) -> ncross, nonherm

Compute `crit` (smallest-magnitude eigenvalue of W⁻¹, scaled by (dV/dψ)²) at step `istep` into
`odet.crit_store`, and count the zero crossings since the previous step (none for steps up to `first_step`):
where the number of negative eigenvalues of W⁻¹ changed, `crossings_through_zero` tells the crossings from the
passages through ∞.

### Returns

  - `ncross::Int`: Zero crossings since the previous step
  - `nonherm::Bool`: True if W⁻¹ was non-Hermitian beyond tolerance
"""
@with_pool pool function check_for_zero_crossings!(odet::OdeState, profiles::Equilibrium.ProfileSplines, istep::Int; first_step::Int=1)

    u = acquire!(pool, eltype(odet.u_store), size(odet.u_store)[1:3])
    psi = odet.psi_store[istep]
    u .= odet.u_store[:, :, :, istep]
    dVdpsi = profiles.dVdpsi_spline(psi; hint=odet.spline_hint)
    crit_val, nonherm = compute_smallest_eigenvalue(u)
    odet.crit_store[istep] = crit_val * dVdpsi^2

    ncross = 0
    if istep > first_step
        u_prev = @view odet.u_store[:, :, :, istep-1]
        if count_negative_wp_inverse(u) != count_negative_wp_inverse(u_prev)
            ncross = crossings_through_zero(u_prev, u)
            ncross != 0 && @info "Zero crossing detected at ψ = $(@sprintf("%.3f", psi))"
        end
    end
    return ncross, nonherm
end

"""
    compute_smallest_eigenvalue(u) -> crit, nonherm

Form W⁻¹ = U₁U₂⁻¹ from the solution `u`, symmetrize it, and return its signed eigenvalue of
smallest magnitude. Also flags whether W⁻¹ was non-Hermitian beyond tolerance before symmetrizing.

### Returns

  - `crit::Float64`: the computed scaled critical eigenvalue
  - `nonherm::Bool`: true if W⁻¹ was non-Hermitian beyond tolerance (> 1e-3)
"""
@with_pool pool function compute_smallest_eigenvalue(u::AbstractArray{ComplexF64,3})

    # In-place wp_inverse = u[:, :, 1] / u[:, :, 2]
    wp_inverse = acquire!(pool, ComplexF64, size(u, 1), size(u, 2))
    U2_tmp = similar!(pool, wp_inverse)
    wp_inverse .= @view u[:, :, 1]
    U2_tmp .= @view u[:, :, 2]
    rdiv!(wp_inverse, lu!(U2_tmp))

    # Measure non-Hermiticity; taking the Hermitian part differs from Hermitian(), which reads one triangle only
    adjoint_wp_inverse = similar!(pool, wp_inverse)
    tmp_mat1 = similar!(pool, wp_inverse)
    tmp_mat2 = similar!(pool, wp_inverse)
    adjoint!(adjoint_wp_inverse, wp_inverse)
    @. tmp_mat1 = 0.5 * (wp_inverse + adjoint_wp_inverse)
    @. tmp_mat2 = 0.5 * (wp_inverse - adjoint_wp_inverse)

    nonherm_error = norm(tmp_mat2) / norm(tmp_mat1)
    nonherm = nonherm_error > 1e-3

    hermitianpart!(wp_inverse)

    # Return the eigenvalue of smallest magnitude, keeping its sign so the caller can detect crossings
    evals = eigvals!(Hermitian(wp_inverse))
    crit = evals[findmin(abs, evals)[2]]
    return crit, nonherm
end

"""
    step_zero_crossings(u, u_prev, nneg_prev, direction) -> ncross, nneg

Zero crossings of a solution over one ODE step from `u_prev` to `u`, taken in ψ direction `direction` (±1). With
F̄ > 0 every crossing moves an eigenvalue of W⁻¹ upward in ψ through 0, so `nneg`, the number of negative eigenvalues
of the congruent U₂ᴴU₁, drops by one. A passage through ∞ (det U₂ = 0) also changes `nneg`; where `nneg` differs from
the previous step's `nneg_prev`, `crossings_through_zero` tells the two apart.
"""
function step_zero_crossings(u::AbstractArray{ComplexF64,3}, u_prev::AbstractArray{ComplexF64,3}, nneg_prev::Int, direction::Int)
    nneg = count_negative_wp_inverse(u)
    ncross = nneg == nneg_prev ? 0 : direction * crossings_through_zero(u_prev, u)
    return ncross, nneg
end

"""
    crossings_through_zero(u_prev, u) -> Int

Net number of eigenvalues of W⁻¹ that pass upward through 0 from `u_prev` to `u` (a downward one counts −1). Each
eigenvalue e sits on the circle at 2 atan(e), where 0 and ±∞ are opposite points. The two sorted sets are matched by
the cyclic shift that moves them least, and a matched pair counts if its shorter arc passes through 0; a passage
through ±∞ takes the arc through π. Unlike the sign of the smallest-|e| eigenvalue, this does not miss a crossing
while another eigenvalue stays closer to 0.
"""
function crossings_through_zero(u_prev::AbstractArray{ComplexF64,3}, u::AbstractArray{ComplexF64,3})
    θ_prev = 2 .* atan.(eigvals!(hermitianpart!(u_prev[:, :, 1] / u_prev[:, :, 2])))
    θ = 2 .* atan.(eigvals!(hermitianpart!(u[:, :, 1] / u[:, :, 2])))
    n = length(θ)
    arc(a, b) = mod(b - a + π, 2π) - π  # signed shorter arc from a to b
    kmax = abs(count(<(0), θ) - count(<(0), θ_prev)) + 1
    k = argmin(k -> sum(i -> abs(arc(θ_prev[mod1(i - k, n)], θ[i])), 1:n), -kmax:kmax)
    return sum(1:n) do i
        a = θ_prev[mod1(i - k, n)]
        b = a + arc(a, θ[i])
        (a < 0 <= b) - (b < 0 <= a)
    end
end

"""
    crossings_beyond_principal(P12, u_s, u_e, direction) -> Int

Zero crossings of a solution in a chunk beyond those of the chunk's principal solution (U₁ = 0 at its start): the
negative eigenvalues of `direction`·herm(X_sᴴ P₁₂⁻¹ X_e), a Morse index split exact for F̄ > 0. `u_s` and `u_e` hold
the solution at the chunk's integration start and end in one column basis, X are their U₁ blocks, and `P12` is the
U₁ block of the principal solution at the end. Each column is first divided by the norm of its `u_s` column, as their
spread near the axis would otherwise turn the small eigenvalues into sign noise. The split is Eq. (5.19) of
arXiv:2107.01928 [Elyseeva, Šepitka & Šimon Hilscher, J. Dyn. Differ. Equ. 35, 2589 (2023)].
"""
function crossings_beyond_principal(P12::AbstractMatrix{ComplexF64}, u_s::AbstractArray{ComplexF64,3}, u_e::AbstractArray{ComplexF64,3}, direction::Int)
    colnorms = [norm(view(u_s, :, j, :)) for j in axes(u_s, 2)]'
    X_s = u_s[:, :, 1] ./ colnorms
    X_e = u_e[:, :, 1] ./ colnorms
    return count_negative_eigenvalues!(hermitianpart!(direction .* (X_s' * (P12 \ X_e))))
end

"""
    count_negative_wp_inverse(u) -> Int

Number of negative eigenvalues of W⁻¹ = U₁U₂⁻¹, from the congruent U₂ᴴU₁.
"""
@with_pool pool function count_negative_wp_inverse(u::AbstractArray{ComplexF64,3})
    h = acquire!(pool, ComplexF64, size(u, 1), size(u, 2))
    mul!(h, view(u, :, :, 2)', view(u, :, :, 1))
    return count_negative_eigenvalues!(Hermitian(h))
end

"""
    count_negative_eigenvalues!(H) -> Int

Number of negative eigenvalues of the Hermitian `H`, from its Bunch–Kaufman LDLᴴ factorization (overwrites `H`): a
1×1 pivot counts by its sign, a 2×2 pivot block holds exactly one. Signs are taken as computed, so unlike `inertia`'s
relative zero tolerance this keeps the small eigenvalues of a graded `H`.
"""
function count_negative_eigenvalues!(H::Hermitian{ComplexF64})
    F = bunchkaufman!(H; check=false)
    nneg, k = 0, 1
    while k <= size(H, 1)
        if F.ipiv[k] < 0  # 2×2 block in rows k, k+1
            nneg += 1
            k += 2
        else
            nneg += real(F.LD[k, k]) < 0
            k += 1
        end
    end
    return nneg
end
