using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs
using CommonRLInterface: reset!, actions, observe, act!, terminated
using Flux
using Plots
using CommonRLInterface.Wrappers: QuickWrapper
using TensorBoardLogger
using Logging
using Statistics

# The following are some basic components needed for DQN

# Override to a discrete action space, and position and velocity observations rather than the matrix.
env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                   observe=mc->observe(mc)[1:2]
                  )

function dqn(env)
    # This network should work for the Q function - an input is a state; the output is a vector containing the Q-values for each action 
    Q = Chain(Dense(2, 512, relu),
              Dense(512, length(actions(env))))

    opt = Flux.setup(ADAM(0.001), Q)

    # set up tensorboard
    lg = TBLogger("tensorboard_logs/dqn_run", min_level=Logging.Info)


    # These were the params I messed with
    gamma = 0.99
    buffer_size = 75000
    batch_size = 128
    update_freq = 1000
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    episode_rewards = []
    buffer = []
    
    reset!(env) # NOTE: after each time the environment reaches a terminal state, you need to reset it

    # this is the fixed target Q network
    Q_target = deepcopy(Q)

    # create your loss function for Q training here
    function loss(Q, s, a_ind, r, sp, done)
        curr = Q(s)[a_ind]

        # correct terminatio
        if done
            target = r
        else
            target = r + 0.99 * maximum(Q(sp))
        end

        return (target - curr)^2
    end

    episode = 0
    steps = 0
    steps_per_ep = 500
    println("Entering training")
    for episode = 1:2000
        reset!(env)
        s = observe(env)
        reward= 0
        episode_loss = []
        q_values = []
        for step = 1:steps_per_ep
            if rand() < epsilon
                a_ind = rand(1:length(actions(env)))
            else
                a_ind = argmax(Q(s))
            end

            a = actions(env)[a_ind]
            r = act!(env, a)
            sp = observe(env)
            done = terminated(env)

            shaped_reward = r

            # I actually don't think I used this to get my best JSON file, did not have as
            # much impact as I expected
            if !done
                shaped_reward += sp[1] * 0.1  # height bonus
                if sp[1] < 0 && sp[2] > 0  # left -> right
                    shaped_reward += sp[2] * 0.05  # more velocity
                elseif sp[1] >= 0 && sp[2] > 0  # right -> left
                    shaped_reward += sp[2] * 0.1  # larger bonus for more velocity in this direction
                end
            end
    
            experience = (s, a_ind, shaped_reward, sp, done) 

            if length(buffer) < buffer_size
                # if length(buffer) % 100 == 0
                #     println("Buffer size: $(length(buffer))")
                # end
                push!(buffer, experience)
            else
                popfirst!(buffer)
            end

            if length(buffer) >= batch_size
                indices = rand(1:length(buffer), batch_size)
                minibatch = buffer[indices]
                Flux.Optimise.train!(loss, Q, minibatch, opt)
            end
            curr_loss = loss(Q, s, a_ind, r, sp, done)
            push!(episode_loss, curr_loss)
            push!(q_values, maximum(Q(s)))

            s = sp
            reward += shaped_reward
            steps += 1

            if step % update_freq == 0
                Q_target = deepcopy(Q)
            end

            if done
                println("DONE, steps = $steps")
                reset!(env)
                break
            end
        end
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode %1 == 0
            println("Ep: $episode, Reward = $reward, epsilon = $epsilon")
        end
        # display(render(env))
        push!(episode_rewards, reward)
        with_logger(lg) do
            @info "training" reward=reward episode=episode avg_loss=mean(episode_loss) max_q=maximum(q_values) min_q=minimum(q_values) avg_q=mean(q_values) epsilon=epsilon
        end
    end

    # ChatGPT for plotting
    p = plot(1:length(episode_rewards), episode_rewards, 
     xlabel="Episode", 
     ylabel="Total Reward", 
     title="DQN Learning Curve",
     legend=false,
     linewidth=2)

    # Add a trend line to better visualize progress
    window_size = min(30, length(episode_rewards))
    moving_avg = [mean(episode_rewards[max(1, i-window_size+1):i]) for i in 1:length(episode_rewards)]
    plot!(p, 1:length(moving_avg), moving_avg, 
        linewidth=3, 
        linestyle=:dash, 
        color=:red, 
        label="Moving Average")

    display(p)
    return Q
end

Q = dqn(env)

HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], "xavier.okeefe@colorado.edu") # you will need to remove the n_episodes=100 keyword argument to create a json file; evaluate needs to run 10_000 episodes to produce a json

#----------
# Rendering
#----------

# You can show an image of the environment like this (use ElectronDisplay if running from REPL):
# display(render(env))

# The following code allows you to render the value function
using Plots
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value")


# function render_value(value)
#     xs = -3.0:0.1:3.0
#     vs = -0.3:0.01:0.3

#     data = DataFrame(
#                      x = vec([x for x in xs, v in vs]),
#                      v = vec([v for x in xs, v in vs]),
#                      val = vec([value([x, v]) for x in xs, v in vs])
#     )

#     data |> @vlplot(:rect, "x:o", "v:o", color=:val, width="container", height="container")
# end

# display(render_value(s->maximum(Q(s))))
