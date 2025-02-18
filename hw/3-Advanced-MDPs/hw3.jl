using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime
using Profile

m = HW3.DenseGridWorld()

function rollout(mdp, policy_function, s0, max_steps=10000)
    r_total = 0.0
    t = 0
    s = s0
    while !isterminal(mdp, s) && t < max_steps
        a= policy_function(mdp, s)
        # println("action $a, state $(s.x), $(s.y)")
        s,r = @gen(:sp, :r)(mdp, s,a)
        r_total+=discount(m)^t*r
        t+=1
    end
    return r_total
end

function heuristic_policy(m, s)

    rewards = [(x, y) for x in 20:20:60 for y in 20:20:60]

    # ChatGPT helped with this syntax
    nearest_idx = argmin([abs(s.x - x) + abs(s.y - y) for (x, y) in rewards])
    target_x, target_y = rewards[nearest_idx]

    if s.x != target_x
        if s.x > target_x
            return :left
        else
            return :right
        end
    end

    if s.y != target_y
        if s.y > target_y
            return :down
        else
            return :up
        end
    end

    return rand(actions(m))
end

# println(actions(m))
# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
# results = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:1000]

# println("MEAN: ", mean(results))
# println("SEM: ", std(results)/sqrt(length(results)))

############
# Question 3
############
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

function (pi::MonteCarloTreeSearch)(s)
    for k in 1:pi.m
        simulate!(pi, s)
    end
    return argmax(a -> pi.Q[(s, a)], actions(pi.P))
end

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d < 1
        return π.U(s)
    end
    # println("SIMULATE")
    P, N, Q, c = π.P, π.N, π.Q, π.c
    𝒜, γ = actions(P), P.discount
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return Float64(π.U(s))
    end
    a = explore(π, s)
    sp, r = @gen(:sp, :r)(P, s, a)

    r = Float64(r)
    # println("Next state: ", sp)the
    # println("Reward: ", r)
    next_q = simulate!(π, sp, d - 1)  # Ensure this is Float64
    q = r + γ * next_q  # This should now be type-stable
    N[(s,a)] +=1
    Q[(s, a)] = get!(Q, (s, a), 0.0) + (q - Q[(s, a)]) / N[(s, a)]
    return q
end


bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns) / Nsa)


function explore(π::MonteCarloTreeSearch, s)
    𝒜, N, Q, c = actions(π.P), π.N, π.Q, π.c
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)]+ c*bonus(N[(s,a)], Ns), 𝒜)
end

m = DenseGridWorld(seed=4)
println(actions(m))
a = :right

@show S = statetype(m)
@show A = actiontype(m)

# These would be appropriate containers for your Q, N, and t dictionaries:
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()

# This is an example state - it is a StaticArrays.SVector{2, Int}
@show s = SA[19,19]

sp, r = @gen(:sp,:r)(m,s,a)
println("S' = $sp")


@show typeof(s)
@assert s isa statetype(m)

# here is an example of how to visualize a dummy tree (q, n, and t should actually be filled in your mcts code, but for this we fill it manually)
# q[(SA[1,1], :right)] = 0.0
# q[(SA[2,1], :right)] = 0.0
# n[(SA[1,1], :right)] = 1
# n[(SA[2,1], :right)] = 0
# t[(SA[1,1], :right, SA[2,1])] = 1


π = MonteCarloTreeSearch(
    P=m, # problem
    N=n, # visit counts
    Q=q, # action value estimates
    d=7, # depth
    m=100, # number of simulations
    c=1.0, # exploration constant
    U=s -> 0.0 # default value function estimate
)
@show best_action = π(s)             # Call the instance like a function

# inchrome(visualize_tree(q, n, t, SA[19,19])) # use inbrowser(visualize_tree(q, n, t, SA[1,1]), "firefox") etc. if you want to use a different browser

############
# Question 4
############

# A starting point for the MCTS select_action function (a policy) which can be used for Questions 4 and 5
function select_action(m, s)
    start = time_ns()
    # S = statetype(m)
    # A = actiontype(m)
    # n = Dict{Tuple{Int, Int}, Int}()  # Preallocate with expected key types
    # q = Dict{Tuple{Int, Int}, Float64}()

    n = Dict{Tuple{S, A}, Int}()
    q = Dict{Tuple{S, A}, Float64}()

    # Define the MCTS policy
    π = MonteCarloTreeSearch(
        P=m, # problem
        N=n, # visit counts
        Q=q, # action value estimates
        d=7, # depth
        m=200, # number of simulations
        c=1.0, # exploration constant
        U=s -> 0.0 # default value function estimate
    )

    # Run MCTS iterations
    for i in 1:1000
        while time_ns() < start + 40_000_000 # uncomment this line to limit the loop to 40ms
            simulate!(π, s)
        end
        # println("iteration $i")
    end

    # Select the best action based on Q values
    best_action = argmax(a -> q[(s::S, a::A)], actions(m))

    # println("Best action $best_action")
    return best_action
end


@profview select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.
@profview select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.

# @btime select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.

# use the code below to evaluate the MCTS policy
# @show results = [rollout(m, select_action, rand(initialstate(m)), 100) for _ in 1:100]

HW3.evaluate(select_action, "xavier.okeefe@colorado.edu")
