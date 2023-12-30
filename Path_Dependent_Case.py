# The unbiased simulation algorithm
# The path dependent case






import numpy as np



# Simulation of d-dimensional Brownian motion
def simulate_brownian_motion(T, N, d):
    if N == 0:
        return np.zeros((1, d))
    dt = T/N  # time increment
    increments = np.sqrt(dt) * np.random.randn(N, d)
    return np.cumsum(increments, axis=0)

# Main algorithm for the path-dependent case
def Path_Dependent_Case(g, mu, sigma, beta, T, d, x0, n):
    # ... (rest of the code as before)

# Parameters
beta = 1.0  # intensity of the Poisson process
T = 1.0     # end time
d = 1       # dimension of Brownian motion
x0 = np.array([0.0])  # initial condition
n = 10      # number of intervals

# Calculate the estimator
Psi_hat = path_dependent_algorithm(g, mu &#8203;``【oaicite:0】``&#8203;
