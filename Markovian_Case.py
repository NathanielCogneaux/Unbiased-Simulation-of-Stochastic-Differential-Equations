# The unbiased simulation algorithm
#
# The Markovian case
# Assumption : - Constant and nondegenerate diffusion coefficient
#              - the drift function μ(t,x) is bounded and continuous in (t,x), uniformly 1/2
#              Hölder in t and uniformly Lipschitz in x, for some constant L > 0

# In this code we use the Hungarian notation as a convention to name the variables


import numpy as np




#We introduce a random discrete time grid with β > 0 a fixed positive constant,
#(τ_i)i>0 be a sequence of i.i.d. E(β)-exponential random variables.
def RandomTimeGrid(nBeta, nSamples, nT):
    # Generate i.i.d. exponential random variables
    arrTau = np.random.exponential(scale=1/beta, size=num_samples)

    # Create the time grid T_k
    arrT_k = np.minimum(np.cumsum(tau), nT)

    # Compute N_t
    nN_t = np.argmax(nT_k >= nT)

    return arrT_k, nN_t



# We now deal with the simulation of the Brownian motion independent of (τi)i>0

# And the Euler scheme of X on the random discrete grid (Tk)k≥0,


def X_EulerScheme(arrX0, nMu, nSigma, arrTimeGrid, nDim):
    # Get the number of steps for the Euler Scheme
    nSteps = len(arrTimeGrid)

    # Initialize array to store X_hat values
    arrX_hat = np.zeros((nSteps, nDim))
    # Set initial value (of dimension d)
    arrX_hat[0] = arrX0

    # Then generate the d-dimensional Brownian motion W and DeltaW
    arrW = np.random.normal(size=(nSteps, nDim)) * np.sqrt(arrTimeGrid)

    arrDeltaW = np.diff(W, axis=0) #axis = 0 calculates the differences between consecutive rows (time steps) of the array

    # Compute DeltaT_k
    arrDeltaT_k = np.diff(arrTimeGrid)

    for k in range(num_steps - 1):
        delta_t = np.diff(time_grid)[k]
        mu_value = mu(time_grid[k], X[k])

        # Euler scheme formula
        X[k + 1] = X[k] + mu_value * delta_t + sigma * np.sqrt(delta_t) * W[k]

    return X

# Example usage:
# Assuming mu, sigma, and time_grid are defined appropriately
x0_value = np.array([0, 0])  # Initial value for X
dimension = 2  # Dimension of the Brownian motion

# Simulate using Euler scheme
resulting_X = simulate_Euler_scheme(x0_value, mu_function, sigma_value, time_grid, dimension)






# Parameters
t_value = 10  # Value of t
beta_value = 0.5  # Beta constant
num_samples_value = 1000  # Number of samples

# Simulate unbiased grid
time_grid, max_index = simulate_unbiased_grid(t_value, beta_value, num_samples_value)

print("Time Grid Tk:", time_grid)
print("Nt (max index where Tk < t):", max_index)

