using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
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
    states = [:Healthy, :InSitu, :Invasive, :Death],
    actions = [:wait, :test, :treat],
    observations = [:positive, :negative],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition = function (s, a)
        if s == :Death
            return Deterministic(:Death)
        elseif s == :Healthy
                return SparseCat([:Healthy, :InSitu], [0.98, 0.02])
        elseif s == :InSitu
            if a == :treat
                return SparseCat([:Healthy, :InSitu], [0.6, 0.4])
            else
                return SparseCat([:InSitu, :Invasive], [0.9, 0.1])
            end
        elseif s == :Invasive
            if a == :treat
                return SparseCat([:Healthy, :Invasive, :Death], [0.2, 0.6, 0.2])
            else
                return SparseCat([:Invasive, :Death], [0.4, 0.6])
            end
        end
    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation = function (s, a, sp)
        if a == :test
            if sp == :Healthy
                return SparseCat([:positive, :negative], [0.05, 0.95])
            elseif sp == :InSitu
                return SparseCat([:positive, :negative], [0.8, 0.2])
            else 
                return Deterministic(:positive)
            end
        elseif a == :treat
            if sp == :InSitu || sp == :Invasive
                return Deterministic(:positive)
            else
                return Deterministic(:negative)
            end
        else
            return Deterministic(:negative)
        end
    end,

    reward = function (s, a)
        if s == :Death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        end
    end,

    initialstate = Deterministic(:Healthy),

    discount = 0.99
)

# evaluate with a random policy
@show POMDPs.actions(tiger)
policy = FunctionPolicy(o->:wait)
sim = RolloutSimulator(max_steps=100)
@show @time mean(POMDPs.simulate(sim, tiger, policy) for _ in 1:10_000)
