using DMUStudent.HW2
using POMDPs: states, actions, transition
using POMDPTools: ordered_states
using LinearAlgebra 
using SparseArrays

function lookahead(prob, T::Dict, R::Dict, U::Vector, s, a, gamma)
    row = T[a][s,:]
    nonzeros = findall(!iszero, row)
    return R[a][s] + gamma * sum(row[nz] * U[nz] for nz in nonzeros)
end

function backup(prob, T::Dict, R::Dict, U::Vector, s, gamma)
    actions = keys(T)

    return maximum(lookahead(prob, T, R, U, s, a, gamma) for a in actions)
end

function value_iteration(prob; gamma=0.95, epsilon=0.001, k_max = 1000)
    T = transition_matrices(prob; sparse=true)
    R = reward_vectors(prob)
    states_set = states(prob)

    V = zeros(length(states_set))
    Vp = ones(length(states_set))
    it = 0

    while norm(V .- Vp, Inf) > epsilon && it < k_max
        V = copy(Vp) # Need to make copy of Vp? 
        Vp = [backup(prob, T, R, V, s, gamma) for s in eachindex(states_set)]
        it = it + 1
        println("it is $it")
    end

    if it == k_max
        println("KMAX")
    else
        println("NOTKMAX")
    end


    return Vp
end


####################################################################################################################################
# Question 3
####################################################################################################################################
V = value_iteration(grid_world)
# println("Final Value Function: ", V)

display(render(grid_world, color=V))



####################################################################################################################################
# Question 4
####################################################################################################################################

function solve(prob; gamma=0.99, epsilon=0.1, k_max=1000)
    T = transition_matrices(prob; sparse=true)  # Use sparse matrices
    R = reward_vectors(prob)
    states_set = states(prob)

    V = zeros(length(states_set))  # Value function
    it = 0

    while true
        Δ = 0  # Track maximum change for convergence
        for s in eachindex(states_set)
            V_old = V[s]
            best_value = -Inf

            # Iterate over actions
            for a in keys(T)
                row = T[a][s, :]  # Sparse row
                indices, values = findnz(row)  # Get nonzero transitions

                # Compute Bellman update efficiently
                value = R[a][s] + gamma * sum(v * V[idx] for (idx, v) in zip(indices, values))

                best_value = max(best_value, value)
            end

            V[s] = best_value
            Δ = max(Δ, abs(V_old - V[s]))
        end

        it += 1
        println("Iteration: $it, Max Change: $Δ")

        if Δ < epsilon || it >= k_max
            break
        end
    end

    return V
end

m = UnresponsiveACASMDP(2)
V = solve(m)
display(@doc(transition_matrices))

@show HW2.evaluate(V)

########
# Extras
########
# using POMDPs: states, stateindex

# s = last(states(m))
# @show si = stateindex(m, s)

# # To convert from a state index to a physical state in the ACAS MDP, use convert_s:
# using POMDPs: convert_s

# @show s = convert_s(ACASState, si, m)

# # To visualize a state in the ACAS MDP, use
# render(m, (s=s,))
