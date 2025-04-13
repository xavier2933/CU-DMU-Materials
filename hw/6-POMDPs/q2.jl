using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW5 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

=#

############
# Question 1
############

# The tiger problem from http://www.sciencedirect.com/science/article/pii/S000437029800023X can be expressed with:

cancer = QuickPOMDP(
    states = [:healthy, :in_situ, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [true, false],
    transition = function (s, a)
        if s == :healthy
            return SparseCat([:healthy, :in_situ], [0.98, 0.02])
        elseif s == :in_situ
            if a == :treat
                return SparseCat([:healthy, :in_situ], [0.6, 0.4])
            else
                return SparseCat([:in_situ, :invasive], [0.9, 0.1])
            end
        elseif s == :invasive
            if a == :treat
                return SparseCat([:healthy, :death, :invasive], [0.2, 0.2, 0.6])
            else
                return SparseCat([:invasive, :death], [0.4, 0.6])
            end
        else
            return Deterministic(:death)
        end
    end,
    observation = function (a, sp)
        if a == :test
            if sp == :healthy
                return SparseCat([true, false], [0.05, 0.95])
            elseif sp == :in_situ
                return SparseCat([true, false], [0.8, 0.2])
            elseif sp == :invasive
                return Deterministic(true)
            end
        elseif a == :treat
            if sp in (:in_situ, :invasive)
                return Deterministic(true)
            end
        end
        return Deterministic(false)
    end,
    reward = function (s, a)
        if s == :death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        end
    end,
    discount = 0.99,
    initialstate = Deterministic(:healthy),
    isterminal = s->s==:death,
)
@assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
# up = HW6Updater(cancer)

# heuristic = FunctionPolicy(function (b)

#                                # Fill in your heuristic policy here
#                                # Use pdf(b, s) to get the probability of a state

#                                return :wait
#                            end
#                           )

# @show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
# @show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
# @show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79

# # evaluate with a random policy
# @show POMDPs.actions(cancer)
# policy = FunctionPolicy(o->:wait)
# sim = RolloutSimulator(max_steps=100)
# @show @time mean(POMDPs.simulate(sim, cancer, policy) for _ in 1:10_000)
