using DMUStudent.HW2
using POMDPs: states, actions
using POMDPTools: ordered_states
using LinearAlgebra

function lookahead(T::Dict, R::Dict, U::Vector, s, a, gamma)
    return R[a][s] + gamma * sum(T[a][s, s′] * U[s′] for s′ in eachindex(U))
end

function backup(T::Dict, R::Dict, U::Vector, s, gamma)
    actions = keys(T)

    return maximum(lookahead(T, R, U, s, a, gamma) for a in actions)
end

function value_iteration(grid_world; gamma=0.95, epsilon=0.01)
    T = transition_matrices(grid_world)
    R = reward_vectors(grid_world)
    states_set = states(grid_world)

    V = zeros(length(states_set))
    Vp = ones(length(states_set))

    while norm(V .- Vp, Inf) > epsilon
        V = copy(Vp) # Need to make copy of Vp? 
        Vp = [backup(T, R, V, s, gamma) for s in eachindex(states_set)]
    end

    return Vp
end


R = reward_vectors(grid_world)

### GPT generated example to help with debugging
s = 1
a = :right
gamma = 0.95
U = rand(length(R[:right]))

lookahead_value = lookahead(T, R, U, s, a, gamma)
println("Lookahead value for state $s taking action $a: $lookahead_value")

"""
Call value iteration fxn, plot
"""
V = value_iteration(grid_world)
println("Final Value Function: ", V)
# POMDPTools.ModelTools.render(HW2.grid_world, color = V)

display(render(grid_world, color=V))

