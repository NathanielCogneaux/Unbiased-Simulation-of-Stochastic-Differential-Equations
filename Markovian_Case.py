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
    arrTau = np.random.exponential(scale=1/nBeta, size=nSamples)

    # Create the time grid (T_k)k≥0
    arrT = np.minimum(np.cumsum(tau), nT)

    # Compute N_t
    nN_t = np.argmax(arrT >= nT) - 1

    return arrT, nN_t


def Unbiased_Simulation_Markovian_Case(arrX0, funcMu, arrSigma, nBeta, nSamples, nT, nDim):
    # Get a random discrete time grid
    arrTimeGrid, nN_t = RandomTimeGrid(nBeta, nSamples, nT)

    # Get the number of steps for the Euler Scheme
    nSteps = len(arrTimeGrid)

    # Initialize array to store X_hat values
    arrX_hat = np.zeros((nSteps, nDim))

    # Set initial value (of dimension d)
    arrX_hat[0] = arrX0

    # We now deal with the simulation of the d-dimensional Brownian motion W independent of (τi)i>0 and DeltaW
    arrW = np.random.normal(size=(nSteps, nDim)) * np.sqrt(arrTimeGrid)
    arrDeltaW = np.diff(W, axis=0) #axis = 0 calculates the differences between consecutive rows (time steps) of the array

    # Compute (DeltaT_k)k≥0
    arrDeltaT = np.diff(arrTimeGrid)

    # Euler scheme loop
    for k in range(nN_t + 1):
        fMuValue_k = funcMu(arrTimeGrid[k], arrX_hat[k])

        # Euler scheme formula
        arrX_hat[k + 1] = arrX_hat[k] + fMuValue_k * arrDeltaT[k + 1] + arrSigma * arrDeltaW[k + 1]

    if nN_t > 0 :
        # Initialize the products of the W^1_k of the estimator
        fProdW1 = 1
        arrSigma_transpose_inv = np.linalg.inv(arrSigma.transpose())
        # W^1_k loop
        for k in range(1, nN_t + 1):
            fProdW1 *= ((funcMu(arrTimeGrid[k], arrX_hat[k]) - funcMu(arrTimeGrid[k-1], arrX_hat[k-1]))*arrSigma_transpose_inv*arrDeltaW[k + 1])/arrDeltaT[k + 1]

    return arrX_hat

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

