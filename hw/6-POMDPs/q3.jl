using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, UnderlyingMDP, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT, QuickMDP
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using DiscreteValueIteration: ValueIterationSolver



m = LaserTagPOMDP()

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
    
    # solve mdp to get decent policy for rollout
    mdp_solver = ValueIterationSolver(max_iterations=1000)
    mdp_policy = solve(mdp_solver, mdp)

    solver = POMCPSolver(tree_queries=1000,
                         c=45.0,
                         max_time = 0.4, # this should be enough time to get a score in the 30s
                         default_action=:measure,
                         estimate_value=FORollout(mdp_policy))
    return solve(solver, m)
end


println("solving pomcp")
pomcp_p = pomcp_solve(m)

println("eval:")
@show HW6.evaluate((pomcp_p, up), n_episodes=100)

# When you get ready to submit, use this version with the full 1000 episodes
# HW6.evaluate((pomcp_p, up), "xavier.okeefe@colorado.edu")

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
