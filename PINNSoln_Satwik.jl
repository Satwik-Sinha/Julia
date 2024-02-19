# Import necessary packages and modules
using NeuralPDE, Lux, Optimization, OptimizationOptimJL,Plots
import ModelingToolkit: Interval

# Define parameters and variables used in the PDE
@parameters t, x
@variables u(..)

# Define the differential operators
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C = 1
eq = Dtt(u(t, x)) ~ C^2 * Dxx(u(t, x))

# Initial and boundary conditions
bcs = [u(t, 0) ~ 0.0,# for all t > 0
    u(t, 1) ~ 0.0,# for all t > 0
    u(0, x) ~ x * (1.0 - x), #for all 0 < x < 1
    Dt(u(0, x)) ~ 0.0] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]
# Discretization
dx = 0.09

# Neural network
dim = 2 # number of dimensions
chain = Lux.Chain(Lux.Dense(dim, 16, Lux.σ), Lux.Dense(16, 16, Lux.σ), Lux.Dense(16, 1))
# Define the discretization method
discretization = PhysicsInformedNN(chain, GridTraining(dx))

# Create the PDE system
@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# Discretize the PDE system
prob = discretize(pde_system, discretization)

# Define the optimization callback function
callback = function (p, l)
    println("Current loss is: $l")
    return false
end

# optimizer
opt = OptimizationOptimJL.BFGS()
res = Optimization.solve(prob, opt; callback = callback, maxiters = 2200)
phi = discretization.phi

using Plots, FileIO

# Define the analytical solution
function u_analytical(t, x)
    return x * (1.0 - x) * (1.0 - cos(C * π * t))
end

# Generate the spatial and temporal grids for the plot
ts = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][2]

# Define the function for the analytical solution
function analytic_sol_func(t, x)
    sum([(8 / (k^3 * pi^3)) * sin(k * pi * x) * cos(C * k * pi * t) for k in 1:2:50000])
end

# Compute the neural network prediction and the analytical solution at the grid points
u_predict = reshape([first(phi([t, x], res.u)) for t in ts for x in xs],(length(ts), length(xs)))
u_real = reshape([analytic_sol_func(t, x) for t in ts for x in xs],(length(ts), length(xs)))

# Compute the absolute difference between the neural network prediction and the analytical solution
diff_u = abs.(u_predict .- u_real)

# Plot the results
using Plots

p1 = plot(xs, ts, u_real, linetype = :contourf, title = "analytic");
p2 = plot(xs, ts, u_predict, linetype = :contourf, title = "predict");
p3 = plot(xs, ts, diff_u, linetype = :contourf, title = "error");
plot(p1, p2, p3)
