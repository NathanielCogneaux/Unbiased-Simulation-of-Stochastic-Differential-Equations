# The unbiased simulation algorithm
#
# The Markovian case
# Assumption : - Constant and nondegenerate diffusion coefficient
#              - the drift function μ(t,x) is bounded and continuous in (t,x), uniformly 1/2
#              Hölder in t and uniformly Lipschitz in x, for some constant L > 0

import numpy as np


#We introduce a random discrete time grid with β > 0 a fixed positive constant,
#(τ_i)i>0 be a sequence of i.i.d. E(β)-exponential random variables.
def RandomTimeGrid(Beta, nSamples, T):
    # Generate i.i.d. exponential random variables
    arrTau = np.random.exponential(scale=1/Beta, size=nSamples)

    # Create the time grid (T_k)k≥0
    arrT = np.minimum(np.cumsum(arrTau), T)

    # Compute N_T
    N_T = np.argmax(arrT >= T)

    return arrT, N_T


def Unbiased_Simulation_Markovian_Case(funcG, arrX0, funcMu, arrSigma, Beta, nSamples, T, nDim):
    # Get a random discrete time grid
    arrTimeGrid, N_T = RandomTimeGrid(Beta, nSamples, T)

    # Get the number of steps for the Euler Scheme
    nSteps = len(arrTimeGrid)

    # Initialize array to store X_hat values
    arrX_hat = np.zeros((nSteps, nDim))

    # Set initial value (of dimension d)
    arrX_hat[0] = arrX0

    # We now deal with the simulation of the d-dimensional Brownian motion W independent of (τi)i>0 and DeltaW
    arrW = np.zeros((nSteps, nDim))
    for i in range(nSteps):
        arrW[i] = np.random.normal(loc=0.0, scale=arrTimeGrid[i], size=nDim)

    arrDeltaW = np.diff(arrW, axis=0) #axis = 0 calculates the differences between consecutive rows (time steps) of the array

    # Compute (DeltaT_k)k≥0
    arrDeltaT = np.diff(arrTimeGrid)

    # Euler scheme loop
    for k in range(N_T + 1):
        MuValue_k = funcMu(arrTimeGrid[k], arrX_hat[k])

        # Euler scheme formula
        arrX_hat[k + 1] = arrX_hat[k] + MuValue_k * arrDeltaT[k + 1] + arrSigma * arrDeltaW[k + 1]

    if N_T > 0 :
        # Initialize the products of the W^1_k of the estimator
        prodW1 = 1

        if nDim > 1:
            arrSigma_transpose_inv = np.linalg.inv(arrSigma.transpose())
        else:
            arrSigma_transpose_inv = 1/arrSigma
        # W^1_k loop
        for k in range(1, N_T + 1):
            prodW1 *= ((funcMu(arrTimeGrid[k], arrX_hat[k]) - funcMu(arrTimeGrid[k-1], arrX_hat[k-1]))*arrSigma_transpose_inv*arrDeltaW[k + 1])/arrDeltaT[k + 1]

        Psi_hat = np.exp(Beta*T)*(funcG(arrX_hat[-1]) - funcG(arrX_hat[N_T]))*Beta**(-N_T)*prodW1

    else :
        Psi_hat = np.exp(Beta*T)*funcG(arrX_hat[-1])

    return Psi_hat
