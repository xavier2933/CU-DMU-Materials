using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, UnderlyingMDP, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT, QuickMDP
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using DiscreteValueIteration: ValueIterationSolver



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

# pomcp_solver = POMCPSolver(
#     c = 1.0, # adjust the exploration parameter just like in MDP MCTS
#     max_time = 0.2, # this should be enough time to get a score in the 30s
    
#     # the most important factor for good MCTS (POMCP) performance is the rollout policy
#     # since every new leaf node in POMCP contains only one state, it makes sense to run a rollout on the fully observable MDP with FO(Fully Observable)Rollout
#     # you can put an MDP policy in here. How would you calculate a good MDP rollout policy? Hint POMDPTools.UnderlyingMDP might help
#     estimate_value = FORollout(FunctionPolicy(s->:right)),
    
#     default_action=:measure # definitely use this in case there is an error half way through evaluation
# )
# A good approach to try is POMCP, implemented in the BasicPOMCP.jl package:
using BasicPOMCP
function pomcp_solve(m) # this function makes capturing m in the rollout policy more efficient

    mdp = UnderlyingMDP(m)
    
    # Solve the MDP to get a good policy for rollouts
    mdp_solver = ValueIterationSolver(max_iterations=100)
    mdp_policy = solve(mdp_solver, mdp)

    solver = POMCPSolver(tree_queries=100,
                         c=50.0,
                         max_time = 0.4, # this should be enough time to get a score in the 30s

                         default_action=:measure,
                         estimate_value=FORollout(mdp_policy))
    return solve(solver, m)
end

#########
# best:

# solver = POMCPSolver(tree_queries=10,
# c=30.0,
# max_time = 0.2, # this should be enough time to get a score in the 30s

# default_action=:measure,
# estimate_value=FORollout(mdp_policy))
# return solve(solver, m)
#####


println("solving pomcp")
pomcp_p = pomcp_solve(m)

println("eval:")
# @show HW6.evaluate((pomcp_p, up), n_episodes=100)

# When you get ready to submit, use this version with the full 1000 episodes
HW6.evaluate((pomcp_p, up), "xavier.okeefe@colorado.edu")

#----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
using POMDPGifs
import Cairo, Fontconfig # needed to display properly

makegif(m, pomcp_p, up, max_steps=30, filename="lasertag.gif")

# You can render a single frame like this
using POMDPTools: stepthrough, render
using Compose: draw, PNG

history = []
for step in stepthrough(m, pomcp_p, up, max_steps=10)
    push!(history, step)
end
displayable_object = render(m, last(history))
# display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
draw(PNG("lasertag.png"), displayable_object)
