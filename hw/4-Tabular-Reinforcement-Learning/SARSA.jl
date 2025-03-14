using POMDPModels: SimpleGridWorld
using LinearAlgebra: I
using CommonRLInterface: render, actions, act!, observe, reset!, AbstractEnv, observations, terminated, clone
import POMDPTools
using SparseArrays
using Statistics: mean
using Plots
using DMUStudent: HW4

####################
# SARSA copied from https://github.com/xavier2933/CU-DMU-Materials/blob/master/notebooks/100-SARSA.jl
####################

function q_learning_episode!(Q, env; ϵ=0.10, γ=0.99, α=0.2)
    start = time()
    
    function policy(s)
        if rand() < ϵ
            return rand(actions(env))
        else
            return argmax(a->Q[(s, a)], actions(env))
        end
    end

    s = observe(env)
    a = policy(s)
    r = act!(env, a)
    sp = observe(env)
    hist = [s]
    cumulative_reward = r

    while !terminated(env)
        # Use max q instead of policy
        max_q = maximum(Q[(sp, ap)] for ap in actions(env))
        Q[(s,a)] += α*(r + γ*max_q - Q[(s, a)])

        s = sp
        a = policy(sp)
        r = act!(env, a)
        cumulative_reward += r
        sp = observe(env)
        push!(hist, sp)
    end

    Q[(s,a)] += α*(r - Q[(s, a)])
    
    # if cumulative_reward >= 5
    #     println("Cumulative reward for episode: $cumulative_reward")
    # end

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function q_learning!(env; n_episodes=100, α=0.2)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, q_learning_episode!(Q, env;
                                          ϵ=max(0.1, 1-i/n_episodes), α=α))
    end
    
    return episodes
end

function sarsa_episode!(Q, env; ϵ=0.10, γ=0.99, α=0.2)
    start = time()
    
    function policy(s)
        if rand() < ϵ
            return rand(actions(env))
        else
            return argmax(a->Q[(s, a)], actions(env))
        end
    end

    s = observe(env)
    a = policy(s)
    r = act!(env, a)
    sp = observe(env)
    hist = [s]
    cumulative_reward = r

    while !terminated(env)
        ap = policy(sp)

        Q[(s,a)] += α*(r + γ*Q[(sp, ap)] - Q[(s, a)])

        s = sp
        a = ap
        r = act!(env, a)
        cumulative_reward += r
        sp = observe(env)
        push!(hist, sp)
    end

    Q[(s,a)] += α*(r - Q[(s, a)])
    # if cumulative_reward >= 5
    #     println("Cumulative reward for episode: $cumulative_reward")
    # end

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function sarsa!(env; n_episodes=100, α=0.2)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        if i % 10_000 == 0
            println("Episode $i")
        end
        reset!(env)
        push!(episodes, sarsa_episode!(Q, env;
                                       ϵ=max(0.1, 1-i/n_episodes), α=α))
    end
    
    return episodes
end

# SARSA-λ implementation
function sarsa_lambda_episode!(Q, env; ϵ=0.05, γ=0.99, α=0.2, λ=0.99)
    start = time()
    
    function policy(s)
        if rand() < ϵ
            return rand(actions(env))
        else
            return argmax(a->Q[(s, a)], actions(env))
        end
    end

    s = observe(env)
    a = policy(s)
    r = act!(env, a)
    sp = observe(env)
    hist = [s]
    N = Dict((s, a) => 0.0)

    while !terminated(env)
        ap = policy(sp)

        N[(s, a)] = get(N, (s, a), 0.0) + 1

        δ = r + γ*Q[(sp, ap)] - Q[(s, a)]

        for ((s, a), n) in N
            Q[(s, a)] += α*δ*n
            N[(s, a)] *= γ*λ
        end

        s = sp
        a = ap
        r = act!(env, a)
        sp = observe(env)
        push!(hist, sp)
    end

    N[(s, a)] = get(N, (s, a), 0.0) + 1
    δ = r - Q[(s, a)]

    for ((s, a), n) in N
        Q[(s, a)] += α*δ*n
        N[(s, a)] *= γ*λ
    end

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function sarsa_lambda!(env; n_episodes=100, kwargs...)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, sarsa_lambda_episode!(Q, env;
                                              ϵ=max(0.01, 1-i/n_episodes),
                                              kwargs...))
    end
    
    return episodes
end

function evaluate(env, policy; n_episodes=1000, max_steps=1000, γ=1.0)
    returns = Float64[]
    for _ in 1:n_episodes
        t = 0
        r = 0.0
        reset!(env)
        s = observe(env)
        while !terminated(env) && t < max_steps
            a = policy(s)
            r += γ^t*act!(env, a)
            s = observe(env)
            t += 1
        end
        push!(returns, r)
    end
    return returns
end

function learning_curve_steps(episodes, env)
	p = plot(xlabel="steps in environment", ylabel="avg return")
	n = 10000
	val = 0
    for (name, eps) in episodes
        println("$name has $(length(eps)) episodes")
        val = length(eps)
    end
    stop = val
    maxval = 0

	for (name, eps) in episodes
	    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
	    xs = [0]
	    ys = [mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env))))]
	    for i in n:n:min(stop, length(eps))
            # println("Processing batch at index $i")
	        newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
	        push!(xs, last(xs) + newsteps)
	        Q = eps[i].Q
            avg = mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env)), n_episodes=1000))
            maxval = max(avg, maxval)
	        push!(ys, avg)
	    end    
	    plot!(p, xs, ys, label=name)
	end
    println("MAX VAL $maxval")
	p
end

function learning_curve_clock(episodes, env)
	p = plot(xlabel="wall clock time", ylabel="avg return")
	n = 10000 
    val = 0
    for (name, eps) in episodes
        println("$name has $(length(eps)) episodes")
        val = length(eps)
    end
    stop = val
	for (name,eps) in episodes
	    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
	    xs = [0.0]
	    ys = [mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env))))]
	    for i in n:n:min(stop, length(eps))
	        newtime = sum(ep.time for ep in eps[i-n+1:i])
	        push!(xs, last(xs) + newtime)
	        Q = eps[i].Q
	        push!(ys, mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env)), n_episodes=1000)))
	    end    
	    plot!(p, xs, ys, label=name)
	end
	p
end

# had claude help with saving plot s
function run_experiment(num_episodes=100_000)
    println("Creating environment...")
    m = HW4.gw
    # m = SimpleGridWorld()
    env = convert(AbstractEnv, m)

    println("Running Q learning with $num_episodes episodes...")
    q_episodes = q_learning!(env, n_episodes=num_episodes, α=0.1)
    
    println("Running SARSA with $num_episodes episodes...")
    sarsa_episodes = sarsa!(env, n_episodes=num_episodes, α=0.1)
    
    # println("Running SARSA-λ with $num_episodes episodes...")
    # lambda_episodes = sarsa_lambda!(env, n_episodes=num_episodes, α=0.1, λ=0.9)
    
    # episodes = Dict("SARSA" => sarsa_episodes, "SARSA-λ" => lambda_episodes, "Q" => q_episodes)

    episodes = Dict("SARSA" => sarsa_episodes, "Q" => q_episodes)

    
    println("Generating learning curve by steps...")
    p1 = learning_curve_steps(episodes, env)
    savefig(p1, "sarsa_learning_curve_steps.png")
    
    println("Generating learning curve by time...")
    p2 = learning_curve_clock(episodes, env)
    savefig(p2, "sarsa_learning_curve_time.png")
    
    println("Done! Plots saved as sarsa_learning_curve_steps.png and sarsa_learning_curve_time.png")
    
    return p1, p2
end


steps_plot, time_plot = run_experiment(500000)

display(steps_plot)
display(time_plot)