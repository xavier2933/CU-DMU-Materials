using DMUStudent.HW2
using POMDPs: states, actions, transition, stateindex
using POMDPTools: ordered_states, POMDPDistributions, Deterministic, render
using LinearAlgebra

function lookahead(prob, R::Dict, U::Vector, s, a, gamma)
    dist = transition(prob, s, a)
    s_idx = stateindex(prob, s)

    # Check if transition is deterministic @chatGPT
    if dist isa Deterministic
        sp = dist.val
        return R[a][s_idx] + gamma * U[stateindex(prob, sp)]
    else
        return R[a][s_idx] + gamma * sum(p * U[stateindex(prob, sp)] for (sp, p) in dist)
    end
end

function backup(prob, R::Dict, U::Vector, s, gamma)
    actionz = actions(prob)

    return maximum(lookahead(prob, R, U, s, a, gamma) for a in actionz)
end

# Old version too slow not used
function value_iteration(prob; gamma=0.95, epsilon=0.01)
    # T = transition_matrices(prob)

    R = reward_vectors(prob)
    states_set = states(prob)

    V = zeros(length(states_set))
    Vp = ones(length(states_set))

    while norm(V .- Vp, Inf) > epsilon
        V = copy(Vp) # Need to make copy of Vp? 
        Vp = [backup(prob, R, V, s, gamma) for s in states(prob)]
        println("iteration")
    end

    return Vp
end

function solve(prob; gamma=0.99, epsilon=0.1, k_max=1000)
    R = reward_vectors(prob)
    states_set = states(prob)

    V = zeros(length(states_set))
    it = 0
    dx = epsilon+1

    while dx > epsilon && it <= k_max
        Vp = copy(V)
        for s in states_set
            s_idx = stateindex(prob, s)

            V[s_idx] = backup(prob, R, V, s, gamma)
        end
        dx = norm(V .- Vp, Inf)
        it += 1
        println("Iteration: $it, Max Change: $dx")
    end

    return V
end

"""
Call funcs
"""
## Q3
V = solve(grid_world;gamma=0.95)

println("Final Value Function: ", V)
display(render(HW2.grid_world, color = V))

## Q4
m = UnresponsiveACASMDP(7)
VP = solve(m)

@show HW2.evaluate(VP, "xavier.okeefe@colorado.edu")