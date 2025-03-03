using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime
using Profile
using JET

struct MonteCarloTreeSearch
    P # problem
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    U # value function estimate
end

function MonteCarloTreeSearch(; P, N, Q, d, m, c, U)
    return MonteCarloTreeSearch(P, N, Q, d, m, c, U)
end

function rollout(mdp, policy_function, s0, max_steps=10000)
    r_total = 0.0
    t = 0
    s = s0
    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s)
        s, r = @gen(:sp, :r)(mdp, s, a)
        r_total += discount(mdp)^t * r
        t += 1
    end
    return Float64(r_total)
end

function heuristic_policy(m, s)
    rewards = [(x, y) for x in 20:20:60 for y in 20:20:60]
    nearest_idx = argmin([abs(s.x - x) + abs(s.y - y) for (x, y) in rewards])
    target_x, target_y = rewards[nearest_idx]

    if s.x != target_x
        return s.x > target_x ? :left : :right
    end

    if s.y != target_y
        return s.y > target_y ? :down : :up
    end

    return rand(actions(m))
end

function random_policy(m,s)
    return rand(actions(m))
end

# Copy pasted some of these from the book, should have changed letters
function simulate!(π::MonteCarloTreeSearch, s, t, d::Int=π.d)
    if d < 1
        # println("Terminating at depth d = $d")
        return Float64(π.U(s))
    end

    P, N, Q, c = π.P, π.N, π.Q, π.c
    𝒜, γ = actions(P), P.discount

    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s, a)] = 0
            Q[(s, a)] = 0.0
        end
    end

    a = explore(π, s)
    sp, r = @gen(:sp, :r)(P, s, a)
    # t[(SA[19,19], :right, SA[20,19])] = 1
    t[(s, a, sp)] = get!(t, (s, a, sp), 0) + 1

    r = Float64(r)
    next_q = simulate!(π, sp, t,d - 1)
    q = r + γ * next_q
    N[(s, a)] += 1
    Q[(s, a)] += (q - Q[(s, a)]) / N[(s, a)]
    return q
end

function bonus(Nsa, Ns)
    return Nsa == 0 ? Inf : sqrt(log(Ns) / Nsa)
end


function explore(π, s)
    𝒜, N, Q, c = actions(π.P), π.N, π.Q, π.c
    Ns = sum(N[(s, a)] for a in 𝒜)
    
    return argmax(a -> Q[(s, a)] + c * bonus(N[(s, a)], Ns), 𝒜)
end

#############
# Question 2
#############

# m = HW3.DenseGridWorld(seed=3)

# println(actions(m))
# # This code runs monte carlo simulations: you can calculate the mean and standard error from the results
# results = [rollout(m, random_policy, rand(initialstate(m))) for _ in 1:1000]

# println("MEAN: ", mean(results))
# println("SEM: ", std(results)/sqrt(length(results)))


#############
# Question 3
##############
# m = HW3.DenseGridWorld(seed=4)

# @show S = statetype(m)
# @show A = actiontype(m)

# n = Dict{Tuple{S, A}, Int}()
# q = Dict{Tuple{S, A}, Float64}()
# t = Dict{Tuple{S, A, S}, Int}()

# π = MonteCarloTreeSearch(
#     P=m, # problem
#     N=n, # visit counts
#     Q=q, # action value estimates
#     d=3, # depth
#     m=7, # number of simulations
#     c=1, # exploration constant
#     U=s -> 0.0 # default value function estimate
# )

# s = SA[19,19]
# for i in 1:π.m
#     # @show t
#     simulate!(π, s, t, π.d)
# end

# inchrome(visualize_tree(q, n, t, s))

# @show maximum(values(q))
# t[(SA[19,19], :right, SA[20,19])] = 1

# println(n)
# println(q)
# println(t)

###############
# Question 4
#################

function select_action(m, s)
    start = time_ns()
    # S = statetype(m)
    # A = actiontype(m)
    # n = Dict{Tuple{Int, Int}, Int}()  # Preallocate with expected key types
    # q = Dict{Tuple{Int, Int}, Float64}()

    n = Dict{Tuple{S, A}, Int}()
    q = Dict{Tuple{S, A}, Float64}()
    t = Dict{Tuple{S, A, S}, Int}()


    # Define the MCTS policy
    π = MonteCarloTreeSearch(
        P=m, # problem
        N=n, # visit counts
        Q=q, # action value estimates
        d=7, # depth
        m=300, # number of simulations
        c=190, # exploration constant
        U=s -> Float64(rollout(m, heuristic_policy, s))  # Ensure Float64 output
        )

    # Run MCTS iterations
    for i in 1:π.m
        # while time_ns() < start + 40_000_000 # uncomment this line to limit the loop to 40ms
        simulate!(π, s, t)
        # end
        # println("iteration $i")
    end

    # Select the best action based on Q values
    # @show maximum(get(q, (s, a), -Inf) for a in actions(m))
    best_action = argmax(a -> get(q, (s, a), -Inf), actions(m))
    # inchrome(visualize_tree(q, n, t, s))

    # println("Best action $best_action")
    return best_action
end



m = HW3.DenseGridWorld()

# @show results = [rollout(m, select_action, rand(initialstate(m)), 100) for _ in 1:100]
# println("MEAN: ", mean(results))
# println("SEM: ", std(results)/sqrt(length(results)))

s = SA[20,24]
# @show rollout(m, select_action, s)
# # inchrome(visualize_tree(q, n, t, s))
# @btime action = select_action(m, s)

# @profview select_action(m, s)
select_action(m, s)
select_action(m, s)

# @report_opt simulate!(π, s, t, 5)



HW3.evaluate(select_action, "xavier.okeefe@colorado.edu", time = true)
