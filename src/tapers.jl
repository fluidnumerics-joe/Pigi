function mkkbtaper(gridspec, ::Type{precision}; alpha=14, threshold=1e-4) where {precision <: AbstractFloat}
    kbnorm = besseli(0, π * alpha)
    # We want the taper to reach threshold at edge of the image
    # This can't be too small, otherwise we encounter significant floating point
    # errors at Float32, but can't be too large or else we truncate the taper too severely
    # (and require significant padding).
    rmax = 0.5
    for r in 0:1e-2:0.5
        if besseli(0, π * alpha * sqrt(1 - 4 * r^2)) / kbnorm < threshold
            rmax = r
            break
        end
    end
    rnorm  = sqrt((gridspec.Nx ÷ 2)^2 + (gridspec.Ny ÷ 2)^2) * gridspec.scalelm / rmax

    taper = ones(precision, gridspec.Nx, gridspec.Ny)

    for lm in CartesianIndices(taper)
        lpx, mpx = Tuple(lm)
        l, m = px2sky(lpx, mpx, gridspec)
        r2 = (l^2 + m^2) / rnorm^2
        if r2 > 0.25
            taper[lm] *= 0
        else
            taper[lm] *= besseli(0, π * alpha * sqrt(1 - 4 * r2)) / kbnorm
        end
    end

    return taper
end

function taperpadding(vimg, vtruncate; alpha=14)
    function kbtaper(r)
        r2 = r^2

        if r2 > 0.25
            return 0.
        else
            return besseli(0, π * alpha * sqrt(1 - 4 * r2)) / besseli(0, π * alpha)
        end
    end

    rimg, rtruncate = 0, 0.5
    for r in range(0, 0.5, length=2000)
        t = kbtaper(r)
        if t > vimg
            rimg = r
        end
        if t > vtruncate
            rtruncate = r
        end
    end

    return rtruncate / rimg
end
