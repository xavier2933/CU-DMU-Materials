using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime
using Profile
struct MonteCarloTreeSearch
    P # problem
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    U # value function estimate
    Ns_cache # Cache for Ns values
end

function MonteCarloTreeSearch(; P, N, Q, d, m, c, U)
    Ns_cache = Dict{statetype(P), Int}()  # Cache for Ns values
    return MonteCarloTreeSearch(P, N, Q, d, m, c, U, Ns_cache)
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
    P, N, Q, c, Ns_cache = π.P, π.N, π.Q, π.c, π.Ns_cache
    𝒜, γ = actions(P), P.discount
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s, a)] = 0
            Q[(s, a)] = 0.0
        end
        Ns_cache[s] = 0
        return π.U(s)
    end
    a = explore(π, s)
    sp, r = @gen(:sp, :r)(P, s, a)
    q = r + γ * simulate!(π, sp, d - 1)
    N[(s, a)] += 1
    Q[(s, a)] += (q - Q[(s, a)]) / N[(s, a)]
    Ns_cache[s] = get(Ns_cache, s, 0) + 1
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns) / Nsa)

function explore(π::MonteCarloTreeSearch, s)
    𝒜, N, Q, c, Ns_cache = actions(π.P), π.N, π.Q, π.c, π.Ns_cache
    Ns = get(Ns_cache, s, sum(N[(s, a)] for a in 𝒜))
    return argmax(a -> Q[(s, a)] + c * bonus(N[(s, a)], Ns), 𝒜)
end

# Example usage
m = DenseGridWorld(seed=4)
S = statetype(m)  # This is a type, e.g., StaticArraysCore.SVector{2, Int64}
A = actiontype(m)  # This is a type, e.g., Symbol

# Initialize dictionaries with the correct key types
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()

# Example state
s = SA[19, 19]  # This is an instance of S

# Define the MCTS policy
π = MonteCarloTreeSearch(
    P=m, # problem
    N=n, # visit counts
    Q=q, # action value estimates
    d=7, # depth
    m=100, # number of simulations
    c=1.0, # exploration constant
    U=s -> 0.0 # default value function estimate
)

# Run MCTS and get the best action
best_action = π(s)
println("Best action: $best_action")

# Function to select the best action within a time limit
function select_action(m, s)
    start = time_ns()
    n = Dict{Tuple{S, A}, Int}()
    q = Dict{Tuple{S, A}, Float64}()

    π = MonteCarloTreeSearch(
        P=m, # problem
        N=n, # visit counts
        Q=q, # action value estimates
        d=7, # depth
        m=100, # number of simulations
        c=1.0, # exploration constant
        U=s -> 0.0 # default value function estimate
    )

    for _ in 1:1000
        while time_ns() < start + 40_000_000 # limit the loop to 40ms
            simulate!(π, s)
        end
    end

    best_action = argmax(a -> q[(s, a)], actions(m))
    println("Best action: $best_action")
    return best_action
end

# @profview select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.
# @btime select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.

# use the code below to evaluate the MCTS policy
# @show results = [rollout(m, select_action, rand(initialstate(m)), 100) for _ in 1:100]

HW3.evaluate(select_action, "xavier.okeefe@colorado.edu")