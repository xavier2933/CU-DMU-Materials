using Plots
using Flux
using Random: randperm
using StaticArrays
using Statistics

# Generate data points

f(x) = (1 - x) * sin(20 * log(x + 0.2))

n = 500
dx = rand(Float32, n)
dy = convert.(Float32, f.(dx))

# Visualize the data
scatter(dx, dy, label="data", title="Sine Function Approximation")

# Define the neural network model
m = Chain(
    Dense(1 => 50, σ),
    Dense(50 => 50, σ),
    Dense(50 => 1)
)

function loss(x, y)
    pred = m(reshape(x, 1, :))
    return sum((pred .- y).^2)
end

lossVec = []
# Prepare data - avoid using SVectors directly in the training loop
data = [(reshape([dx[i]], 1, 1), [dy[i]]) for i in 1:length(dx)]

# Training loop
for i in 1:12000
    Flux.train!(loss, Flux.params(m), data, Flux.Optimise.Adam())   
    currLoss = sum([loss(d[1], d[2]) for d in data])
    push!(lossVec, currLoss)
    if i % 50 == 0
        sorted_x = sort(dx)
        predictions = []
        for x in sorted_x
            pred = m(reshape([x], 1, 1))[1]
            push!(predictions, pred)
        end
        
        p = plot(sorted_x, x -> (1 - x) * sin(20 * log(x + 0.2)), label="sin(4π x)", lw=2)
        plot!(p, sorted_x, predictions, label="NN approximation", lw=2)
        scatter!(p, dx, dy, label="data", alpha=0.6)

        p2 = plot(1:i, lossVec, label="Total Loss", 
        xlabel="Iterations", ylabel="Loss", 
        title="Learning Curve", linewidth=2)

        display(p2)
        
        display(i)
        display(p)
    end

    if currLoss < 0.01
        break
    end
end

# Generate 100 random test points
n_test = 100
test_x = rand(Float32, n_test)
pred_y = [m(reshape([x],1,1))[1] for x in test_x]

true_y = f.(test_x)

## ChatGPT helped with plotting
# Create a smooth curve for plotting the original function
x_smooth = range(0, 1, length=500)
y_smooth = f.(x_smooth)

# Create the plot
p4 = plot(x_smooth, y_smooth, 
          label="Original function", 
          linewidth=2, 
          title="Neural Network Predictions vs Original Function",
          xlabel="x", 
          ylabel="y")

# Add the predicted points as a scatter plot
scatter!(p4, test_x, pred_y, 
         label="NN predictions", 
         alpha=0.7, 
         markersize=4, 
         markershape=:circle)
