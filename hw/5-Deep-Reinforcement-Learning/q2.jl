using Plots
using Flux
using Random: randperm
using StaticArrays

# Generate data points

f(x) = (1 - x) * sin(20 * log(x + 0.2))

n = 200
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

# Prepare data - avoid using SVectors directly in the training loop
data = [(reshape([dx[i]], 1, 1), [dy[i]]) for i in 1:length(dx)]

# Training loop
for i in 1:8000
    Flux.train!(loss, Flux.params(m), data, Flux.Optimise.Adam())   
    if i % 50 == 0
        sorted_x = sort(dx)
        predictions = []
        for x in sorted_x
            pred = m(reshape([x], 1, 1))[1]
            push!(predictions, pred)
        end
        
        p = plot(sorted_x, x -> sin(4*pi*x), label="sin(4π x)", lw=2)
        plot!(p, sorted_x, predictions, label="NN approximation", lw=2)
        scatter!(p, dx, dy, label="data", alpha=0.6)
        
        display(i)
        display(p)
    end
end