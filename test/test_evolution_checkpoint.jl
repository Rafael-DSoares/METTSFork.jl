using Test
using HDF5
using ITensors
using ITensorMPS

@testset "evolution_checkpoint.jl" begin
    N = 8

    function heisenberg(sites)
        terms = OpSum()
        for j in 1:(length(sites) - 1)
            terms += 0.5, "S+", j, "S-", j + 1
            terms += 0.5, "S-", j, "S+", j + 1
            terms += 1.0, "Sz", j, "Sz", j + 1
        end
        return MPO(terms, sites)
    end

    # Every job of a chain builds new site indices, so each call gets its own
    new_run() = begin
        sites = siteinds("S=1/2", N; conserve_qns=true)
        H = heisenberg(sites)
        psi0 = MPS(sites, [isodd(j) ? "Up" : "Dn" for j in 1:N])
        (; sites, H, psi0)
    end

    params = Dict{String,Any}("N" => N, "beta" => 2.0, "state" => [isodd(j) ? 1 : 2 for j in 1:N])
    fidelity(a, b) = abs(inner(a, replace_siteinds(b, siteinds(a))))

    # Resumes after every single step: a deadline in the past stops each call after one step
    function resume_step_by_step(evolve, filename)
        calls = 0
        while true
            run = new_run()
            checkpoint = EvolutionCheckpoint(filename, params, run.sites; deadline=-Inf)
            calls += 1
            try
                return evolve(run, checkpoint), run, calls
            catch e
                e isa EvolutionInterrupted || rethrow()
                @test isfile(filename)
            end
        end
    end

    kwargs = (tau=0.2, cutoff=1e-10, maxm=8, tau0=0.05, nsubdiv=2, kkrylov=2, silent=true)

    @testset "timeevo_tdvp_extend" begin
        evolve(run, checkpoint) =
            timeevo_tdvp_extend(run.H, run.psi0, -1.1; checkpoint=checkpoint, kwargs...)

        ref = new_run()
        psi_ref, lognorm_ref = timeevo_tdvp_extend(ref.H, ref.psi0, -1.1; kwargs...)

        # A checkpoint that is never written changes nothing
        mktempdir() do dir
            filename = joinpath(dir, "checkpoint.hdf5")
            run = new_run()
            psi, lognorm = evolve(run, EvolutionCheckpoint(filename, params, run.sites; interval=Inf))
            @test !isfile(filename)
            @test fidelity(psi_ref, psi) ≈ 1 atol = 1e-10
            @test lognorm ≈ lognorm_ref atol = 1e-10
        end

        mktempdir() do dir
            filename = joinpath(dir, "checkpoint.hdf5")
            (psi, lognorm), run, calls = resume_step_by_step(evolve, filename)
            # nsubdiv extension steps, 5 steps of 0.2 and the remainder 0.05; the last call
            # only returns the state the previous one stopped after
            @test calls == 2 + 5 + 1 + 1
            @test siteinds(psi) == run.sites
            @test fidelity(psi_ref, psi) ≈ 1 atol = 1e-10
            @test lognorm ≈ lognorm_ref atol = 1e-10
            @test read_checkpoint_params(filename) == params
        end
    end

    @testset "timeevo_tdvp_extend_measurements" begin
        measure(run) = psi -> Dict{String,Float64}("energy" => real(inner(psi', run.H, psi)))
        collapse_func = psi -> [expect(psi, "Sz"; sites=j) > 0 ? 1 : 2 for j in 1:length(psi)]
        evolve(run, checkpoint) = timeevo_tdvp_extend_measurements(
            run.H, run.psi0, 0.3, 1.0, 0.6, measure(run), collapse_func;
            checkpoint=checkpoint, kwargs...)

        ref = new_run()
        psi_ref, lognorm_ref, meas_ref, collapsed_ref, times_ref = timeevo_tdvp_extend_measurements(
            ref.H, ref.psi0, 0.3, 1.0, 0.6, measure(ref), collapse_func; kwargs...)

        mktempdir() do dir
            filename = joinpath(dir, "checkpoint.hdf5")
            (psi, lognorm, meas, collapsed, times), run, calls = resume_step_by_step(evolve, filename)
            # nsubdiv extension steps, tau0 -> tau, 2 steps to t_collapse, 2 steps to t_max
            @test calls == 2 + 1 + 2 + 2 + 1
            @test length(times) == 4
            @test fidelity(psi_ref, psi) ≈ 1 atol = 1e-10
            @test lognorm ≈ lognorm_ref atol = 1e-10
            @test times ≈ times_ref
            @test keys(meas) == keys(meas_ref)
            for k in keys(meas_ref)
                @test meas[k] ≈ meas_ref[k] atol = 1e-10
            end
            @test collapsed == collapsed_ref
        end
    end

    # The gauge of the stored state must survive the resume: the evolution after it
    # (in particular a basis extension) depends on where the orthogonality center is
    @testset "resume keeps the orthogonality center" begin
        mktempdir() do dir
            filename = joinpath(dir, "checkpoint.hdf5")
            run = new_run()
            psi = orthogonalize(run.psi0, 3)
            METTS.save_evolution_checkpoint(EvolutionCheckpoint(filename, params, run.sites), 1, psi, 0.0)

            other = new_run()
            checkpoint = EvolutionCheckpoint(filename, params, other.sites)
            state = METTS.resume_checkpoint(METTS.checkpoint_tracker(checkpoint))
            @test siteinds(state.psi) == other.sites
            @test ITensorMPS.ortho_lims(state.psi) == 3:3
        end
    end

    @testset "refuses a checkpoint written with other params" begin
        mktempdir() do dir
            filename = joinpath(dir, "checkpoint.hdf5")
            run = new_run()
            @test_throws EvolutionInterrupted timeevo_tdvp_extend(run.H, run.psi0, -1.1;
                checkpoint=EvolutionCheckpoint(filename, params, run.sites; deadline=-Inf), kwargs...)

            other = merge(params, Dict{String,Any}("beta" => 3.0))
            @test_throws r"'beta' = 3.0 in this run but 2.0" timeevo_tdvp_extend(run.H, run.psi0, -1.1;
                checkpoint=EvolutionCheckpoint(filename, other, run.sites), kwargs...)

            missing_key = Dict{String,Any}(k => v for (k, v) in params if k != "N")
            @test_throws r"'N'" timeevo_tdvp_extend(run.H, run.psi0, -1.1;
                checkpoint=EvolutionCheckpoint(filename, missing_key, run.sites), kwargs...)
        end
    end
end
