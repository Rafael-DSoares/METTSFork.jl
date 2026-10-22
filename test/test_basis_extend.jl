using Test
using ITensors
using ITensorMPS

@testset "basis_extend.jl" begin
    # Heisenberg model on a 4x2 triangular cylinder; bonds of range up to 3 in the MPS ordering
    N = 8
    bonds = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5), (3, 6), (4, 5),
             (4, 6), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)]
    sites = siteinds("S=1/2", N; conserve_qns=true)
    terms = OpSum()
    for (a, b) in bonds
        terms += 0.5, "S+", a, "S-", b
        terms += 0.5, "S-", a, "S+", b
        terms += 1.0, "Sz", a, "Sz", b
    end
    H = MPO(terms, sites)
    extend(psi) = METTS.basis_extend(psi, H; extension_krylovdim=3, extension_cutoff=1e-12)
    deviation(a, b) = norm(a - b) / norm(b)

    # The two extensions of timeevo_tdvp_extend, with one 1TDVP sweep and the in-place
    # orthogonalize! of entropy_von_neumann in between. With a relative cutoff in `extend`,
    # the second extension changed psi completely for 34 of these 256 product states.
    states = [[(n >> (j - 1)) & 1 + 1 for j in 1:N] for n in 0:(2^N - 1)]
    worst_first = worst_second = 0.0
    for state in states
        psi = MPS(sites, state)
        psix = extend(psi)
        worst_first = max(worst_first, deviation(psix, psi))

        phi = tdvp(H, -0.01, psix; updater_backend="applyexp", nsweeps=1, nsite=1,
            cutoff=1e-12, maxdim=16, normalize=true)
        orthogonalize!(phi, N ÷ 2)
        worst_second = max(worst_second, deviation(extend(phi), phi))
    end
    @test worst_first < 1e-10
    @test worst_second < 1e-10
end
