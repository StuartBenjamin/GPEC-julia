"""
    extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64) -> (r, z, ν)

Extract plasma surface geometry from equilibrium at specified flux coordinate.

Evaluates equilibrium bicubic spline around the flux surface to get R, Z coordinates
and computes the toroidal angle offset ν for vacuum calculations.

## Arguments

  - `equil`: Equilibrium solution with rzphi bicubic spline
  - `ψ`: Normalized flux coordinate (0 at axis, 1 at edge)

## Returns

Tuple of:

  - `r::Vector{Float64}`: R-coordinates around flux surface [mtheta]
  - `z::Vector{Float64}`: Z-coordinates around flux surface [mtheta]
  - `ν::Vector{Float64}`: Toroidal angle offset ν [mtheta]

## Implementation

The equilibrium bicubic spline `rzphi` stores:

  - rzphi.f[1] = r² (or rfac²)
  - rzphi.f[2] = offset (straight-fieldline poloidal angle offset from geometric poloidal angle)
  - rzphi.f[3] = ν (toroidal angle offset from geometric toroidal angle)
  - rzphi.f[4] = jac (Jacobian)

From these we compute:

  - r_minor = √(rzphi.f[1])
  - θ_cyl = 2π*(θ_sfl + rzphi.f[2])
  - R = R₀ + r_minor * cos(θ_cyl)
  - Z = Z₀ + r_minor * sin(θ_cyl)

## Reference

[Chance Phys. Plasmas 1997 2161]
Matches GPEC's ahg_write and gpvacuum_flxsurf approach (gpvacuum.f line 291-296)
"""
function extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64)

    # Allocate output arrays
    mtheta = length(equil.rzphi_ys)
    r_minor = zeros(mtheta)
    θ_cyl = zeros(mtheta)
    ν = zeros(mtheta)

    # Evaluate equilibrium around the flux surface
    hint2d = (Ref(1), Ref(1))
    for (i, θ_sfl) in enumerate(equil.rzphi_ys)
        # Get minor radius, geometric poloidal angle, and toroidal angle offset
        r_minor[i] = sqrt(equil.rzphi_rsquared((ψ, θ_sfl); hint=hint2d))
        θ_cyl[i] = 2π * (θ_sfl + equil.rzphi_offset((ψ, θ_sfl); hint=hint2d))
        ν[i] = equil.rzphi_nu((ψ, θ_sfl); hint=hint2d)
    end

    # Compute R and Z using the computed cylindrical coordinates
    r = equil.ro .+ r_minor .* cos.(θ_cyl)
    z = equil.zo .+ r_minor .* sin.(θ_cyl)
    return r, z, ν
end

"""
    distribute_to_equal_arc_grid(xin, zin)

Given a set of points (xin, zin) that define a closed curve in 2D, redistribute these points to be equally spaced
in terms of arc length along the curve. This is useful for ensuring that points on a wall or plasma surface are
uniformly distributed in space rather than in the parameter θ. This performs the same function as eqarcw in the
Fortran code. We now use FastInterpolations instead of a manual Lagrange interpolation.

# Arguments

  - `xin::Vector{Float64}`: x-coordinates of the original points defining the curve (endpoint not included)
  - `zin::Vector{Float64}`: z-coordinates of the original points defining the curve (endpoint not included)

# Returns

  - `xout::Vector{Float64}`: x-coordinates of the redistributed points, equally spaced in arc length
  - `zout::Vector{Float64}`: z-coordinates of the redistributed points, equally spaced in arc length
"""
function distribute_to_equal_arc_grid(xin::Vector{Float64}, zin::Vector{Float64})

    mtheta = length(xin)
    dθ = 2π / mtheta
    θ_grid = range(; start=0, length=mtheta, step=dθ)

    # Build periodic splines for derivatives on the closed loop
    spline_x = cubic_interp(θ_grid, xin; bc=PeriodicBC(; endpoint=:exclusive))
    spline_z = cubic_interp(θ_grid, zin; bc=PeriodicBC(; endpoint=:exclusive))

    # Calculate cumulative arc length using numerical integration
    arc_length = zeros(mtheta + 1)
    for i in 1:mtheta
        # Use a mid-point derivative approximation
        theta_mid = θ_grid[i] + dθ / 2.0
        d_xin = spline_x(theta_mid; deriv=DerivOp(1))
        d_zin = spline_z(theta_mid; deriv=DerivOp(1))
        # Accumulate length: ds = (ds/dt) * dt
        ds_dθ = sqrt(d_xin^2 + d_zin^2)
        arc_length[i+1] = arc_length[i] + ds_dθ * dθ
    end

    # Re-parameterize based on equal arc-length segments by interpolate the original (x,z) data at the equal arc length points
    arc_length_targets = range(; start=0, length=mtheta, step=arc_length[end]/mtheta)
    # arc_length is non-uniform, so close the loop and use periodic_cubic_interp
    xout = periodic_cubic_interp(arc_length, [xin; xin[1]]).(arc_length_targets)
    zout = periodic_cubic_interp(arc_length, [zin; zin[1]]).(arc_length_targets)
    return xout, zout
end

"""
    resample_closed_contour(xin, zin, m) -> (xout, zout)

Resample a closed contour (endpoint not repeated) onto `m` points equally spaced in chord
length, using periodic cubic splines. Starts at the first input point and keeps its orientation.
"""
function resample_closed_contour(xin::Vector{Float64}, zin::Vector{Float64}, m::Int)
    n = length(xin)
    s = zeros(n + 1)
    for i in 1:n
        j = i == n ? 1 : i + 1
        s[i+1] = s[i] + hypot(xin[j] - xin[i], zin[j] - zin[i])
    end
    targets = range(; start=0, length=m, step=s[end] / m)
    # s is non-uniform: close the loop and use periodic_cubic_interp
    return periodic_cubic_interp(s, [xin; xin[1]]).(targets), periodic_cubic_interp(s, [zin; zin[1]]).(targets)
end

"""
    wall_clearance(xp, zp, xw, zw) -> (n_outside, min_gap)

Count plasma points `(xp, zp)` outside the closed wall polygon `(xw, zw)` (even-odd rule) and
return the minimum plasma-to-wall-segment distance.
"""
function wall_clearance(xp::AbstractVector, zp::AbstractVector, xw::AbstractVector, zw::AbstractVector)
    nw = length(xw)
    n_outside = 0
    min_gap = Inf
    for (x, z) in zip(xp, zp)
        inside = false
        for i in 1:nw
            j = i == nw ? 1 : i + 1
            x1, z1, dx, dz = xw[i], zw[i], xw[j] - xw[i], zw[j] - zw[i]
            ((z1 > z) != (zw[j] > z)) && x < x1 + (z - z1) * dx / dz && (inside = !inside)
            seg2 = dx^2 + dz^2
            t = seg2 > 0 ? clamp(((x - x1) * dx + (z - z1) * dz) / seg2, 0.0, 1.0) : 0.0
            min_gap = min(min_gap, hypot(x - x1 - t * dx, z - z1 - t * dz))
        end
        inside || (n_outside += 1)
    end
    return n_outside, min_gap
end

"""
    periodic_wrap(x, n) -> Int

Inline function for fast periodic wrapping for indices near the valid range [1, n].
Equivalent to `mod1(x, n)` but avoids division for small offsets.
Only valid when `x` is within one period of the valid range (i.e., `1-n < x < 2n`).
"""
@inline periodic_wrap(x::Int, n::Int) = x < 1 ? x + n : (x > n ? x - n : x)

"""
    cross3!(c, a, b, idx)

Inline function for fast cross product of two 3D vectors at a given index.

# Arguments

  - `c::AbstractMatrix`: Output matrix
  - `a::AbstractMatrix`: First input vector
  - `b::AbstractMatrix`: Second input vector
  - `idx`: Index of the output vector
"""
@inline function cross3!(
    c::AbstractMatrix,
    a::AbstractMatrix,
    b::AbstractMatrix,
    idx::Int
)
    @inbounds begin
        a1 = a[idx, 1]
        a2 = a[idx, 2]
        a3 = a[idx, 3]
        b1 = b[idx, 1]
        b2 = b[idx, 2]
        b3 = b[idx, 3]

        c[idx, 1] = a2*b3 - a3*b2
        c[idx, 2] = a3*b1 - a1*b3
        c[idx, 3] = a1*b2 - a2*b1
    end
end
