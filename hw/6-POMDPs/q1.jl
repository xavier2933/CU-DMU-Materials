using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, alpha_vector, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater, Policies
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using Plots
using StatsPlots


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

# chatGPT to plot as well
function plot_alpha_vectors(policy, pomdp; title_prefix="")
    # Get state space
    S = ordered_states(pomdp)
    
    # For Tiger POMDP, we can parameterize the belief space by 
    # the probability that the tiger is on the left
    b_left = 0:0.01:1
    
    # Map action indices to names
    action_names = ["Listen", "Open Left", "Open Right"]
    
    p = plot(; 
        xlabel="P(Tiger on Left)", 
        ylabel="Expected Value", 
        title="$(title_prefix) Alpha Vectors for Tiger POMDP",
        legend=:outertopright
    )
    
    # Get alphas and actions based on policy type
    if policy isa HW6AlphaVectorPolicy
        alphas = policy.alphas
        actions = policy.alpha_actions
    elseif policy isa Policies.AlphaVectorPolicy
        alphas = policy.alphas
        actions = [policy.action_map[i] for i in 1:length(policy.alphas)]
    else
        error("Unsupported policy type: $(typeof(policy))")
    end
    
    for (i, alpha) in enumerate(alphas)
        # For Tiger, belief vector is [p(left), p(right)] = [p, 1-p]
        values = [alpha[1] * p + alpha[2] * (1-p) for p in b_left]
        
        # Get action name safely
        action_idx = actions[i]
        action_label = if 1 <= action_idx <= length(action_names)
            action_names[action_idx]
        else
            "Action $action_idx"
        end
        
        plot!(p, b_left, values, label=action_label)
    end
    
    # Also plot the upper envelope (which defines the value function)
    upper_envelope = [maximum([alpha[1] * p + alpha[2] * (1-p) for alpha in alphas]) for p in b_left]
    plot!(p, b_left, upper_envelope, label="Value Function", linewidth=2, color=:black, linestyle=:dash)
    
    return p
end

function compare_alpha_vectors(qmdp_policy, sarsop_policy, pomdp)
    # Create a layout with 2 plots
    p = plot(
        layout=(2,1), 
        size=(800, 1000), 
        legend=:outertopright
    )
    
    # Plot QMDP alpha vectors on the first subplot
    qmdp_plot = plot_alpha_vectors(qmdp_policy, pomdp; title_prefix="QMDP")
    
    # Plot SARSOP alpha vectors on the second subplot
    sarsop_plot = plot_alpha_vectors(sarsop_policy, pomdp; title_prefix="SARSOP")
    
    # Combine plots
    combined_plot = plot(qmdp_plot, sarsop_plot, layout=(2,1), size=(800, 1000))
    
    return combined_plot
end


m = TigerPOMDP()

qmdp_p = qmdp_solve(m)

# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
# comparison = compare_alpha_vectors(qmdp_p, sarsop_p, m)
# display(comparison)
up = HW6Updater(m)

# @show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
# @show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)

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
            else
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

@show qmdp_p = qmdp_solve(cancer)
@show qmdp_p.alphas
println("SARSOP")
@show sarsop_p = solve(SARSOPSolver(), cancer)

policy = qmdp_p
sim = RolloutSimulator()
up = DiscreteUpdater(cancer)

println("QMDP")
@show @time mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)
println("SARSOP")

@show @time mean(POMDPs.simulate(sim, cancer, sarsop_p, up) for _ in 1:1000)

# comparison = compare_alpha_vectors(qmdp_p, sarsop_p, cancer)
# state_list = POMDPs.states(cancer)

# # Extract alpha vectors from the policy
# alphas = qmdp_p.alphas
# actions = qmdp_p.alpha_actions

# # Create a grouped bar plot
# p = groupedbar(hcat(alphas...)', 
#                xticks=(1:length(state_list), string.(state_list)),
#                label=reshape(string.(actions), 1, :),
#                title="Alpha Vectors from QMDP Solution",
#                xlabel="States",
#                ylabel="Value",
#                legend=:topright,
#                bar_width=0.7,
#                lw=0,
#                alpha=0.8,
#                framestyle=:box,
#                fontfamily="Computer Modern")

# display(p)

# state_list_s = POMDPs.states(cancer)

# # Extract alpha vectors from the policy
# alphas_s = sarsop_p.alphas
# actions_s = sarsop_p.action_map

# # Create a grouped bar plot
# p_s = groupedbar(hcat(alphas...)', 
#                xticks=(1:length(state_list), string.(state_list)),
#                label=reshape(string.(actions), 1, :),
#                title="Alpha Vectors from SARSOP Solution",
#                xlabel="States",
#                ylabel="Value",
#                legend=:topright,
#                bar_width=0.7,
#                lw=0,
#                alpha=0.8,
#                framestyle=:box,
#                fontfamily="Computer Modern")

# display(p_s)

