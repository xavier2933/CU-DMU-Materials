using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime

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

println(actions(m))
# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
results = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:1000]

println("MEAN: ", mean(results))
println("SEM: ", std(results)/sqrt(length(results)))