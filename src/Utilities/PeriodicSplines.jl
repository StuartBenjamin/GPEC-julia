using FastInterpolations: cubic_interp, Series, CubicFit

"""
    periodic_cubic_interp(xs, fs; nwrap=16)

Periodic cubic spline through `fs` (a vector, or a matrix fitted as a `Series` by rows) on a possibly non-uniform
grid `xs` that includes the periodic endpoint (`xs[end] - xs[1]` is the period, `fs[end] == fs[1]`).
FastInterpolations' `PeriodicBC` is wrong on non-uniform grids (≤ 0.4.19: `_build_periodic_cache` shifts the upper
diagonal by one cell), giving O(h³) values and O(h²) slopes. Instead `nwrap` cells are wrapped onto each side and a
`CubicFit` spline is fitted, whose end effects decay as (2 - √3)^nwrap.
"""
function periodic_cubic_interp(xs::AbstractVector{Float64}, fs::AbstractVecOrMat{Float64}; nwrap::Int=16)
    n = length(xs) - 1
    k = min(nwrap, n)
    period = xs[end] - xs[1]
    xe = [xs[(n-k+1):n] .- period; xs; xs[2:(k+1)] .+ period]
    return cubic_interp(xe, _wrap_periodic(fs, n, k); bc=CubicFit())
end

_wrap_periodic(fs::AbstractVector, n::Int, k::Int) = [fs[(n-k+1):n]; fs; fs[2:(k+1)]]
_wrap_periodic(fs::AbstractMatrix, n::Int, k::Int) = Series([fs[(n-k+1):n, :]; fs; fs[2:(k+1), :]])
