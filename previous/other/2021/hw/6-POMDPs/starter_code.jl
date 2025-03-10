using POMDPs
using DMUStudent.HW6
using POMDPModelTools: transition_matrices, reward_vectors, SparseCat, Deterministic
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP
using POMDPSimulators: RolloutSimulator
using BeliefUpdaters: DiscreteBelief
using SARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using POMDPPolicies: FunctionPolicy

##################
# Problem 1: Tiger
##################

#--------
# Updater
#--------

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    bp_vec = zeros(length(states(up.m)))
    bp_vec[1] = 1.0

    # Fill in code for belief update
    # Note that the ordering of the entries in bp_vec must be consistent with stateindex(m, s) (the container returned by states(m) does not necessarily obey this order)

    return DiscreteBelief(up.m, bp_vec)
end

# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example
# Z(o | a, s') can be programmed with
Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.

# This is needed to automatically turn any distribution into a discrete belief.
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

# Note: to check your belief updater code, you can use BeliefUpdaters: DiscreteUpdater. It should function exactly like your updater.

#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)

    # Fill in code to choose action based on alpha vectors

    return first(actions(b.pomdp))
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order

#------
# QMDP
#------

function qmdp_solve(m, discount=discount(m))

    # Fill in VI code for QMDP

    acts = actiontype(m)[]
    alphas = Vector{Float64}[]
    for a in actions(m)

        # Fill in alpha vector calculation
        # Note that the ordering of the entries in the alpha vectors must be consistent with stateindex(m, s) (the container returned by states(m) does not necessarily obey this order)
        
    end
    return HW6AlphaVectorPolicy(alphas, acts)
end

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

@show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
@show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)

###################
# Problem 2: Cancer
###################

cancer = QuickPOMDP(

    # Fill in your actual code from last homework here

    states = [:healthy, :in_situ, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [true, false],
    transition = (s, a) -> Deterministic(s),
    observation = (a, sp) -> Deterministic(false),
    reward = (s, a) -> 0.0,
    discount = 0.99,
    initialstate = Deterministic(:death),
    isterminal = s->s==:death,
)

@assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up = HW6Updater(cancer)

heuristic = FunctionPolicy(function (b)

                               # Fill in your heuristic policy here

                               return :wait
                           end
                          )

@show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
@show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
@show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79

#####################
# Problem 3: LaserTag
#####################

m = LaserTagPOMDP()

qmdp_p = qmdp_solve(m)
up = HW6Updater(m)

# to display what's going on, use the following code
using BeliefUpdaters: DiscreteUpdater
using ElectronDisplay
using POMDPSimulators: stepthrough
using POMDPModelTools: render

up = DiscreteUpdater(m) # you may want to replace this with your updater to test it
for step in stepthrough(m, qmdp_p, up, "a,r,sp,o,bp"; max_steps=10)
    electrondisplay(render(m, step))
end

@show HW6.evaluate((qmdp_p, up)) # 27.44
