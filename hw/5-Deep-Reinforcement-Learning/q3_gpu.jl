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
using CUDA

# The following are some basic components needed for DQN

# Override to a discrete action space, and position and velocity observations rather than the matrix.
env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                   observe=mc->observe(mc)[1:2]
                  )
function dqn(env)
    # Check if CUDA is available and use GPU if it is
    use_gpu = try
        CUDA.functional()
        CUDA.allowscalar(true)
    catch
        false
    end
    
    device = use_gpu ? gpu : cpu
    println("Using device: ", use_gpu ? "GPU" : "CPU")
    
    # This network should work for the Q function - an input is a state; the output is a vector containing the Q-values for each action 
    Q = Chain(Dense(2, 128, relu),
                Dense(128, length(actions(env)))) |> device

    opt = Flux.setup(ADAM(0.0005), Q)
    lg = TBLogger("tensorboard_logs/dqn_run", min_level=Logging.Info)

    gamma = 0.99
    buffer_size = 50000
    batch_size = 128
    update_freq = 200
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.98

    episode_rewards = []
    buffer = []
    
    reset!(env) # NOTE: after each time the environment reaches a terminal state, you need to reset it

    # this is the fixed target Q network
    Q_target = deepcopy(Q)

    # Modified loss function to avoid scalar indexing on GPU
    function compute_loss(s, a_ind, r, sp, done)
        # Move data to appropriate device
        s_tensor = use_gpu ? reshape(Float32.(s), :, 1) |> gpu : s
        sp_tensor = use_gpu ? reshape(Float32.(sp), :, 1) |> gpu : sp
        
        # Forward pass to get all Q values
        current_q_values = Q(s_tensor)
        
        # Move to CPU to extract the relevant Q value
        if use_gpu
            current_q = Array(current_q_values)[a_ind]
        else
            current_q = current_q_values[a_ind]
        end
        
        # Calculate target Q value
        target = if done
            r
        else
            next_q_values = Q_target(sp_tensor)
            # Move to CPU to find maximum
            next_q_max = use_gpu ? maximum(Array(next_q_values)) : maximum(next_q_values)
            r + gamma * next_q_max
        end
        
        # Return squared error
        return (target - current_q)^2
    end

    episode = 0
    steps = 0
    steps_per_ep = 300
    println("Entering training")
    for episode = 1:75
        reset!(env)
        s = observe(env)
        reward = 0
        episode_loss = []
        q_values = []
        for step = 1:steps_per_ep
            # For action selection
            s_tensor = use_gpu ? reshape(Float32.(s), :, 1) |> gpu : s
            
            if rand() < epsilon
                a_ind = rand(1:length(actions(env)))
            else
                # Get Q values and move to CPU for argmax
                q_vals = use_gpu ? Array(Q(s_tensor)) : Q(s_tensor)
                a_ind = argmax(q_vals)
            end

            a = actions(env)[a_ind]
            r = act!(env, a)
            sp = observe(env)
            done = terminated(env)

            # Store experience in CPU memory
            experience = (s, a_ind, r, sp, done)
            if length(buffer) < buffer_size
                push!(buffer, experience)
            else
                # Replace random experience
                idx = rand(1:buffer_size)
                buffer[idx] = experience
            end

            if length(buffer) >= batch_size
                # Train on a batch
                indices = rand(1:length(buffer), batch_size)
                batch_loss = 0.0
                
                # Custom batch training loop to avoid scalar indexing
                for i in indices
                    exp = buffer[i]
                    # Gradients and update for this experience
                    gs = Flux.gradient(Q) do q
                        loss_val = compute_loss(exp[1], exp[2], exp[3], exp[4], exp[5])
                        batch_loss += loss_val
                        return loss_val
                    end
                    Flux.Optimisers.update!(opt, Q, gs[1])
                end
                
                push!(episode_loss, batch_loss/batch_size)
            end
            
            # For logging current Q values
            curr_loss = compute_loss(s, a_ind, r, sp, done)
            push!(episode_loss, curr_loss)
            
            # Get max Q value for logging
            q_vals = use_gpu ? Array(Q(s_tensor)) : Q(s_tensor)
            push!(q_values, maximum(q_vals))

            s = sp
            reward += r
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

        if episode % 1 == 0
            println("Ep: $episode, Reward = $reward, epsilon = $epsilon")
        end
        
        push!(episode_rewards, reward)
        with_logger(lg) do
            @info "training" reward=reward episode=episode avg_loss=mean(episode_loss) max_q=maximum(q_values) min_q=minimum(q_values) avg_q=mean(q_values) epsilon=epsilon
        end
    end
    # Return the model on CPU for evaluation
    return use_gpu ? cpu(Q) : Q
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
