using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions


m = LaserTagPOMDP()


function qmdp_solve(m, discount=discount(m))

    # Fill in Value Iteration to compute the Q-values
    S = ordered_states(m)
    A = ordered_actions(m)
    V = zeros(length(S))

    for _ in 1:1000
        prev = copy(V)

        for (i, s) in enumerate(S)
            max_q = -Inf
            for a in A
                q_sa = reward(m,s,a)

                for sp in S
                    p_sp = pdf(transition(m,s,a),sp)
                    ip = stateindex(m, sp)
                    q_sa += discount * p_sp * prev[ip]
                end
                max_q = max(max_q, q_sa)
            end
            V[i] = max_q
        end

        if maximum(abs.(V-prev)) < 1e-6
            break
        end
    end


    acts = actiontype(m)[]
    alphas = Vector{Float64}[]
    for a in A
        push!(acts, a)

        a_vec = zeros(length(S))

        for (i, s) in enumerate(S)
            a_vec[i] = reward(m,s,a)

            for sp in S
                p_sp = pdf(transition(m,s,a), sp)
                sp_i = stateindex(m, sp)
                a_vec[i] += discount * p_sp * V[sp_i]
            end
        end
        # Fill in alpha vector calculation
        # Note that the ordering of the entries in the alpha vectors must be consistent with stateindex(m, s) (states(m) does not necessarily obey this order, but ordered_states(m) does.)
        push!(alphas, a_vec)
    end
    return HW6AlphaVectorPolicy(alphas, acts)
end

println("Solving")
# qmdp_p = qmdp_solve(m)
up = DiscreteUpdater(m) # you may want to replace this with your updater to test it

# Use this version with only 100 episodes to check how well you are doing quickly
# @show HW6.evaluate((qmdp_p, up), n_episodes=100)

# A good approach to try is POMCP, implemented in the BasicPOMCP.jl package:
using BasicPOMCP
function pomcp_solve(m) # this function makes capturing m in the rollout policy more efficient
    solver = POMCPSolver(tree_queries=10,
                         c=1.0,
                         default_action=first(actions(m)),
                         estimate_value=FORollout(FunctionPolicy(s->rand(actions(m)))))
    return solve(solver, m)
end
println("solving pomcp")
pomcp_p = pomcp_solve(m)

println("eval:")
@show HW6.evaluate((pomcp_p, up), n_episodes=100)

# When you get ready to submit, use this version with the full 1000 episodes
# HW6.evaluate((qmdp_p, up), "REPLACE_WITH_YOUR_EMAIL@colorado.edu")

#----------------
# Visualization
# (all code below is optional)
#----------------

# # You can make a gif showing what's going on like this:
# using POMDPGifs
# import Cairo, Fontconfig # needed to display properly

# makegif(m, qmdp_p, up, max_steps=30, filename="lasertag.gif")

# # You can render a single frame like this
# using POMDPTools: stepthrough, render
# using Compose: draw, PNG

# history = []
# for step in stepthrough(m, qmdp_p, up, max_steps=10)
#     push!(history, step)
# end
# displayable_object = render(m, last(history))
# # display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
# draw(PNG("lasertag.png"), displayable_object)
