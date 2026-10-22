using Printf

""" computes the number of steps, and the remainder of the time step if not commensurate """
function n_steps_remainder(time::Number, tau::Real)
    n_steps = floor(abs(time) / tau)
    time_tau = n_steps * tau * time / abs(time)
    remainder = time - time_tau
    return n_steps, remainder, time_tau
end

function entropy_von_neumann(psi::MPS, b::Int)
    nrm = norm(psi)
    s = siteinds(psi)
    orthogonalize!(psi, b)
    _, S = svd(psi[b], (linkind(psi, b - 1), s[b]))
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = (S[n, n] / nrm)^2
        SvN -= p * log(p)
    end
    return SvN
end

"""
Basic time evolution using TDVP

starting out with 2TDVP until a maximal bond dimension is achieved
then it switches to 1TDVP
a commensurate step size is automatically chosen

With `checkpoint` set, the evolution can be stopped and resumed (see `EvolutionCheckpoint`).
"""
function timeevo_tdvp(H::MPO, psi0::MPS, time::Number;
    tau::Number=0.1, cutoff::Float64=1e-6,
    maxm::Int64=1000, normalize::Bool=true,
    silent=false, solver_backend::AbstractString="applyexp",
    shift::Real=0., checkpoint::Union{Nothing,EvolutionCheckpoint}=nothing)

    tracker = checkpoint_tracker(checkpoint)
    state = resume_checkpoint(tracker)
    if isnothing(state)
        psi, log_norm, done = copy(psi0), 0.0, 0
    else
        psi, log_norm, done = state.psi, state.log_norm, state.step
    end

    return timeevo_tdvp_steps(H, psi, time, log_norm, 0, done, tracker;
        tau=tau, cutoff=cutoff, maxm=maxm, silent=silent, solver_backend=solver_backend)
end

"""
Steps of `timeevo_tdvp`, starting from `psi` with accumulated `log_norm`. The steps are
numbered `offset + 1, offset + 2, ...` for checkpointing; those up to number `done` are
skipped, because a resumed `psi` already contains them.
"""
function timeevo_tdvp_steps(H::MPO, psi::MPS, time::Number, log_norm::Real,
    offset::Integer, done::Integer, tracker::Union{Nothing,CheckpointTracker};
    tau::Number, cutoff::Float64, maxm::Int64, silent, solver_backend::AbstractString)
    N = length(psi)
    n_steps, remainder, time_tau = n_steps_remainder(time, tau)
    n_steps = Int(n_steps)

    # One sweep of size dt: 2TDVP until the maximal bond dimension is reached, then 1TDVP
    function sweep(dt, tau_printed)
        nsite = maxlinkdim(psi) < maxm ? 2 : 1
        t = @elapsed begin
            psi = tdvp(H, dt, psi;
                updater_backend=solver_backend,
                nsweeps=1,
                nsite=nsite,
                cutoff=cutoff,
                maxdim=maxm,
                normalize=false)

            log_norm += 2.0*log(norm(psi))
            normalize!(psi)
        end
        svn = entropy_von_neumann(psi, N ÷ 2)
        linkdim = maxlinkdim(psi)
        if !silent
            @printf("    %dTDVP sweep, tau: %.5f, SvN: %.4f, maxm: %6d, time: %.5f secs\n",
                nsite, tau_printed, svn, linkdim, t)
            flush(stdout)
        end
        GC.gc()
    end

    for step in 1:n_steps
        offset + step <= done && continue
        sweep(time_tau / n_steps, tau)
        checkpoint_step!(tracker, offset + step, psi, log_norm)
    end
    if !isapprox(remainder, 0; rtol=1e-6, atol=1e-6) && offset + n_steps + 1 > done
        sweep(remainder, abs(remainder))
        checkpoint_step!(tracker, offset + n_steps + 1, psi, log_norm)
    end

    return psi, log_norm
end


"""
Time evolution using TDVP with intitial basis extension

starting out with 2TDVP until a maximal bond dimension is achieved
then it switches to 1TDVP
a commensurate step size is automatically chosen

With `checkpoint` set, the evolution can be stopped and resumed (see `EvolutionCheckpoint`).
"""
function timeevo_tdvp_extend(H::MPO, psi0::MPS, time::Number;
    tau::Number=0.1, cutoff::Float64=1e-6,
    maxm::Int64=1000, tau0::Float64=0.05,
    nsubdiv::Int64=4, kkrylov::Int64=3,
    normalize::Bool=true, silent=false,
    solver_backend::AbstractString="applyexp",
    shift::Real=0., checkpoint::Union{Nothing,EvolutionCheckpoint}=nothing)

    log_norm = 0.0

    N = length(psi0)
    tau_init = min(tau0, abs(time))
    time_init = time / abs(time) * tau_init

    # distribute intial times logarithmically
    times_init = [time_init / (2^(nsubdiv - 1))]
    for exp in (nsubdiv-1):-1:1
        append!(times_init, time_init / (2^exp))
    end

    psi = copy(psi0)

    # Steps are numbered over the basis extension (1:nsubdiv) and the bulk evolution, so that
    # a resumed run skips exactly the steps contained in the checkpoint
    tracker = checkpoint_tracker(checkpoint)
    done = 0
    state = resume_checkpoint(tracker)
    if !isnothing(state)
        psi, log_norm, done = state.psi, state.log_norm, state.step
    end

    for isub in 1:nsubdiv
        isub <= done && continue
        l1 = maxlinkdim(psi)
        t = @elapsed begin
            println("    Performing basis extension ...")
            flush(stdout)
            psi = basis_extend(psi, H; extension_krylovdim=kkrylov,
                extension_cutoff=1e-12)
        end
        l2 = maxlinkdim(psi)
        if !silent
            @printf("    Basis extension, maxm: %4d -> %4d, time: %.5f secs\n",
                l1, l2, t)
        end
        t = @elapsed begin
            # psi = tdvp(H, psi, times_init[isub];
            #            nsweeps=1, nsite=1, cutoff=1e-16,
            #            maxdim=maxm, normalize=normalize,
            #            solver_backend=solver_backend, shift=shift)
            psi = tdvp(H, times_init[isub], psi;
                updater_backend=solver_backend,
                nsweeps=1,
                nsite=1,
                cutoff=cutoff,
                maxdim=maxm,
                normalize=false)

            log_norm += 2.0*log(norm(psi))
            normalize!(psi)
        end

        svn = entropy_von_neumann(psi, N ÷ 2)
        linkdim = maxlinkdim(psi)
        if !silent
            @printf("    1TDVP sweep, tau: %.5f, SvN: %.4f, maxm: %6d, time: %.5f secs\n",
                abs(times_init[isub]), svn, linkdim, t)
            flush(stdout)
        end
        checkpoint_step!(tracker, isub, psi, log_norm)
    end
    GC.gc()

    # Perform the bulk time evolution
    time_bulk = time - time_init
    psi, log_norm = timeevo_tdvp_steps(H, psi, time_bulk, log_norm, nsubdiv, done, tracker;
        tau=tau, cutoff=cutoff, maxm=maxm, silent=silent, solver_backend=solver_backend)

    return psi, log_norm
end
