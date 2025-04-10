using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
# using POMDPPolicies

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
    S = b.state_list

    # Fill in code for belief update
    # Note that the ordering of the entries in bp_vec must be consistent with stateindex(m, s) (the container returned by states(m) does not necessarily obey this order)
    for (ip, sp) in enumerate(S)
        po = pdf(observation(up.m, a, sp), o)
        bp_vec[ip] = po * sum(pdf(transition(up.m, s, a), sp) * b.b[i] for (i, s) in enumerate(S))
    end
    if sum(bp_vec) ≈ 0
        fill!(bp_vec, 1)
    end
    return DiscreteBelief(up.m, bp_vec./sum(bp_vec))
end

# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example if you want to use more mathematical-looking functions, you could use the following:
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

# Note: to check your belief updater code, you can use POMDPTools: DiscreteUpdater. It should function exactly like your updater.

#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    # Get belief as a vector
    
    # Calculate value for each alpha vector (dot product)
    values = [sum(alpha .* b.b) for alpha in p.alphas]
    
    # Choose action with highest value
    best_idx = argmax(values)
    return p.alpha_actions[best_idx]
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order

#------
# QMDP
#------

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

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

@show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
@show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)

# Add this code to compare your updater with DiscreteUpdater
# used claude to test these
function compare_updaters()
    # Create the POMDP
    tiger = TigerPOMDP()
    
    # Create both updaters
    hw6_updater = HW6Updater(tiger)
    std_updater = DiscreteUpdater(tiger)
    
    # Initialize beliefs from the same distribution
    initial_dist = initialstate(tiger)
    hw6_b = initialize_belief(hw6_updater, initial_dist)
    std_b = initialize_belief(std_updater, initial_dist)
    
    # Print initial beliefs
    println("Initial beliefs:")
    println("HW6Updater: ", hw6_b.b)
    println("DiscreteUpdater: ", std_b.b)
    println("Match: ", all(isapprox.(hw6_b.b, std_b.b, atol=1e-10)))
    println()
    
    # Test sequence of actions and observations
    action_obs_sequence = [
        (TIGER_LISTEN, true),
        (TIGER_LISTEN, false),
        (TIGER_OPEN_LEFT, false),
        (TIGER_LISTEN, true),
        (TIGER_LISTEN, true),
        (TIGER_OPEN_LEFT, false),
        (TIGER_OPEN_LEFT, true)


    ]
    
    for (i, (a, o)) in enumerate(action_obs_sequence)
        # Update both beliefs
        hw6_b = update(hw6_updater, hw6_b, a, o)
        std_b = update(std_updater, std_b, a, o)
        
        # Print and compare
        println("After update $i (action: $a, observation: $o):")
        println("HW6Updater: ", hw6_b.b)
        println("DiscreteUpdater: ", std_b.b)
        println("Match: ", all(isapprox.(hw6_b.b, std_b.b, atol=1e-10)))
        println()
    end
    
    return hw6_b, std_b
end

# Run the comparison
# hw6_final, std_final = compare_updaters()