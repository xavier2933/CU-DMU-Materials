using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPModelTools: Deterministic, Uniform, SparseCat
using POMDPPolicies: FunctionPolicy
using BeliefUpdaters: DiscreteUpdater
using POMDPSimulators: RolloutSimulator
using Statistics: mean
import POMDPs

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW5 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

=#

############
# Question 1
############

# The tiger problem from http://www.sciencedirect.com/science/article/pii/S000437029800023X can be expressed with:

tiger = QuickPOMDP(
    states = [:TL, :TR],
    actions = [:OL, :OR, :L],
    observations = [:TL, :TR],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition = function (s, a)
        if a == :L
            return Deterministic(s)
        else
            return Uniform([:TL, :TR])
        end
    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation = function (s, a, sp)
        if a == :L
            if sp == :TL
                return SparseCat([:TL, :TR], [0.85, 0.15])
            else
                return SparseCat([:TR, :TL], [0.85, 0.15])
            end
        else
            return Uniform([:TL, :TR])
        end
    end,

    reward = function (s, a)
        if a == :L
            return -1.0
        elseif a == :OL
            if s == :TR
                return 10.0
            else
                return -100.0
            end
        else # a = :OR
            if s == :TL
                return 10.0
            else
                return -100.0
            end
        end
    end,

    initialstate = Uniform([:TL, :TR]),

    discount = 0.95
)

# evaluate with a random policy
policy = FunctionPolicy(o->rand(POMDPs.actions(tiger)))
sim = RolloutSimulator(max_steps=100)
updater = DiscreteUpdater(tiger)
@show @time mean(POMDPs.simulate(sim, tiger, policy, updater) for _ in 1:10_000)

############
# Question 2
############

# The notebook at https://github.com/zsunberg/CU-DMU-Materials/blob/master/notebooks/08_Neural_Networks.ipynb can serve as a starting point for this problem.

############
# Question 3
############

using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper
using VegaLite
using ElectronDisplay: electrondisplay
using DataFrames: DataFrame

# The following are some basic components needed for DQN

# Since the mc environment has a continuous action space and DQN uses a discrete action space, you can choose a subset of the actions to use and create an environment with an overridden action space:
env = QuickWrapper(HW5.mc, actions=[-1.0, 0.0, 1.0])

# This network should work for the Q function - an input is a state; the output is a vector containing the Q-values for each action 
Q = Chain(Dense(2, 64, relu),
          Dense(64, length(actions(env))))

# We can create 1 tuple of experience like this
s = observe(env)
a_ind = 1 # action index - this, rather than the actual action, will be needed in the loss function
r = act!(env, actions(env)[a_ind])
sp = observe(env)
done = terminated(env)

experience_tuple = (s, a_ind, r, sp, done)

# this container should work well for the experience buffer:
buffer = [experience_tuple]
# you will need to push more experience into it and randomly select data for training

# create your loss function for Q training here
function loss(s, a_ind, r, sp, done)
    return (r-Q(s)[a_ind])^2 # this is not correct! you need to replace it with the true Q-learning loss function
    # make sure to take care of cases when the problem has terminated correctly
end

# do your training like this (you may have to adjust some things, and you will have to do this many times):
data = rand(buffer, 10)
Flux.Optimise.train!(loss, params(Q), data, ADAM(0.001))

HW5.evaluate(s->actions(env)[argmax(Q(s))])

#----------
# Rendering
#----------

# You can show an image of the environment like this:
electrondisplay(render(env), focus=false)

# The following code allows you to render the value function using VegaLite and ElectronDisplay

function render_value(value)
    xs = -3.0:0.1:3.0
    vs = -0.3:0.01:0.3

    data = DataFrame(
                     x = vec([x for x in xs, v in vs]),
                     v = vec([v for x in xs, v in vs]),
                     val = vec([value([x, v]) for x in xs, v in vs])
    )

    data |> @vlplot(:rect, "x:o", "v:o", color=:val, width="container", height="container")
end

electrondisplay(render_value(s->maximum(Q(s))), focus=false)
