using Plots

x = 1:10
y = sin.(x)

plot(x, y, 
    title="sin plot",
    xlabel="x",
    ylabel="y",
    linewidth=2,
    marker=:circle)