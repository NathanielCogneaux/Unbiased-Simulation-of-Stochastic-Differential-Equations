# The unbiased simulation algorithm
#
# The Markovian case
# Assumption : - Constant and nondegenerate diffusion coefficient
#              - the drift function μ(t,x) is bounded and continuous in (t,x), uniformly 1/2
#              Hölder in t and uniformly Lipschitz in x, for some constant L > 0

import numpy as np




#We introduce a random discrete time grid with β > 0 a fixed positive constant,
#(τ_i)i>0 be a sequence of i.i.d. E(β)-exponential random variables.

def simulate_grid(beta, num_samples, T):
    # Generate i.i.d. exponential random variables
    tau = np.random.exponential(scale=1/beta, size=num_samples)

    # Create the time grid Tk
    tk = np.cumsum(tau)

    # Calculate Nt
    n_t = np.argmax(tk >= t)

    return tk, n_t

# Parameters
t_value = 10  # Value of t
beta_value = 0.5  # Beta constant
num_samples_value = 1000  # Number of samples

# Simulate unbiased grid
time_grid, max_index = simulate_unbiased_grid(t_value, beta_value, num_samples_value)

print("Time Grid Tk:", time_grid)
print("Nt (max index where Tk < t):", max_index)

