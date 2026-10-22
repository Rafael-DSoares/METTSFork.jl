using HDF5
using Printf

"""
    EvolutionCheckpoint(filename, params, sites; interval=3600.0, deadline=deadline_from_env())

Lets a long imaginary-time evolution be stopped and resumed by a later job. Passed as the
`checkpoint` keyword to `timeevo_tdvp`, `timeevo_tdvp_extend` or
`timeevo_tdvp_extend_measurements`, it makes the evolution

- resume from `filename` if the file exists. The `params` stored in it must equal `params`
  (otherwise an error is thrown), and the stored MPS gets the site indices `sites` of the
  current run;
- write `filename` every `interval` seconds of wall time;
- write `filename` and throw `EvolutionInterrupted` when the next step might not finish before
  `deadline` (seconds since the epoch; by default the environment variable `METTS_DEADLINE`).

`params` holds everything that defines the evolution (couplings, truncation, step sizes, initial
state, ...), so that a checkpoint is never continued with other settings. Values must be
numbers, strings or arrays of numbers.

The file is replaced atomically, so a job killed while writing leaves the previous checkpoint
intact. The caller removes the file once the result of the evolution has been stored.
"""
struct EvolutionCheckpoint
    filename::String
    params::Dict{String,Any}
    sites::Vector
    interval::Float64
    deadline::Float64
end

EvolutionCheckpoint(filename::AbstractString, params::AbstractDict, sites;
    interval::Real=3600.0, deadline::Real=deadline_from_env()) =
    EvolutionCheckpoint(String(filename), Dict{String,Any}(params), collect(sites),
        Float64(interval), Float64(deadline))

""" deadline in seconds since the epoch from the environment variable METTS_DEADLINE, Inf if unset """
deadline_from_env() = haskey(ENV, "METTS_DEADLINE") ? parse(Float64, ENV["METTS_DEADLINE"]) : Inf

"""
    EvolutionInterrupted(filename)

Thrown when an evolution with an `EvolutionCheckpoint` stops because its deadline is near.
The state has been written to `filename`; calling the evolution again with the same
checkpoint continues from there.
"""
struct EvolutionInterrupted <: Exception
    filename::String
end

Base.showerror(io::IO, e::EvolutionInterrupted) =
    print(io, "EvolutionInterrupted: deadline near, state written to $(e.filename)")

# Safety margin before the deadline: the next step may take 1.5 times the longest step seen,
# plus this many seconds for writing the checkpoint and the caller's own work.
const CHECKPOINT_STOP_MARGIN = 900.0

# Longest step of any evolution in this process. Kept across calls, so that a new evolution
# started shortly before the deadline (e.g. the next METTS step) already knows how long its
# bulk steps will take.
const longest_step_seconds = Ref(0.0)

mutable struct CheckpointTracker
    checkpoint::EvolutionCheckpoint
    last_save::Float64
    last_step_end::Float64
end

checkpoint_tracker(::Nothing) = nothing
checkpoint_tracker(checkpoint::EvolutionCheckpoint) =
    CheckpointTracker(checkpoint, time(), time())

"""
    read_checkpoint_params(filename) -> Dict{String,Any}

Returns the `params` stored in an evolution checkpoint file.
"""
read_checkpoint_params(filename::AbstractString) = h5open(read_params, filename, "r")

read_params(f) = Dict{String,Any}(k => read(f["params"], k) for k in keys(f["params"]))

function check_checkpoint_params(checkpoint::EvolutionCheckpoint, stored::AbstractDict)
    for k in union(keys(stored), keys(checkpoint.params))
        haskey(stored, k) ||
            error("'$k' is missing in checkpoint '$(checkpoint.filename)'; it was written by another kind of run.")
        haskey(checkpoint.params, k) ||
            error("Checkpoint '$(checkpoint.filename)' has '$k', which this run does not set; it was written by another kind of run.")
        stored[k] == checkpoint.params[k] ||
            error("'$k' = $(checkpoint.params[k]) in this run but $(stored[k]) in checkpoint '$(checkpoint.filename)'.")
    end
end

function save_evolution_checkpoint(checkpoint::EvolutionCheckpoint, step::Integer, psi::MPS,
    log_norm::Real; times=nothing, measurements=nothing, collapsed_state=nothing)
    tmp = checkpoint.filename * ".tmp"
    h5open(tmp, "w") do f
        write(f, "psi", psi)
        f["step"] = step
        f["log_norm"] = log_norm
        f["max_step_seconds"] = longest_step_seconds[]
        for (k, v) in checkpoint.params
            f["params/$k"] = v
        end
        # Empty arrays are not written; they are restored as empty on loading
        if !isnothing(times) && !isempty(times)
            f["times"] = times
        end
        if !isnothing(measurements)
            for (k, v) in measurements
                isempty(v) || (f["measurements/$k"] = v)
            end
        end
        if !isnothing(collapsed_state)
            f["collapsed_state"] = collapsed_state
        end
    end
    mv(tmp, checkpoint.filename; force=true)
end

"""
Loads the checkpoint of `tracker` if its file exists. Returns `nothing` otherwise.
"""
function resume_checkpoint(tracker::Union{Nothing,CheckpointTracker})
    isnothing(tracker) && return nothing
    checkpoint = tracker.checkpoint
    isfile(checkpoint.filename) || return nothing
    state = h5open(checkpoint.filename, "r") do f
        check_checkpoint_params(checkpoint, read_params(f))
        (psi=replace_siteinds(read(f, "psi", MPS), checkpoint.sites),
            step=read(f, "step"),
            log_norm=read(f, "log_norm"),
            max_step_seconds=read(f, "max_step_seconds"),
            times=haskey(f, "times") ? read(f, "times") : Float64[],
            measurements=haskey(f, "measurements") ?
                Dict{String,Vector{Float64}}(k => read(f["measurements"], k) for k in keys(f["measurements"])) :
                Dict{String,Vector{Float64}}(),
            collapsed_state=haskey(f, "collapsed_state") ? read(f, "collapsed_state") : nothing)
    end
    longest_step_seconds[] = max(longest_step_seconds[], state.max_step_seconds)
    println("    Resuming from checkpoint $(checkpoint.filename) after step $(state.step)")
    flush(stdout)
    tracker.last_step_end = time()
    return state
end

"""
Called after every step of an evolution (`step` counts all steps of the evolution). Writes the
checkpoint when the save interval has passed or the deadline is near, and in the latter case
throws `EvolutionInterrupted`.
"""
checkpoint_step!(::Nothing, step, psi, log_norm; kwargs...) = nothing

function checkpoint_step!(tracker::CheckpointTracker, step::Integer, psi::MPS, log_norm::Real; kwargs...)
    checkpoint = tracker.checkpoint
    now = time()
    longest_step_seconds[] = max(longest_step_seconds[], now - tracker.last_step_end)

    stop = checkpoint.deadline - now < 1.5 * longest_step_seconds[] + CHECKPOINT_STOP_MARGIN
    if stop || now - tracker.last_save >= checkpoint.interval
        t = @elapsed save_evolution_checkpoint(checkpoint, step, psi, log_norm; kwargs...)
        @printf("    Checkpoint after step %d written to %s, time: %.5f secs\n",
            step, checkpoint.filename, t)
        flush(stdout)
        tracker.last_save = time()
        stop && throw(EvolutionInterrupted(checkpoint.filename))
    end
    tracker.last_step_end = time()
    return nothing
end
