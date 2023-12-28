# The unbiased simulation algorithm
#
# The Markovian case
# Assumption : - Constant and nondegenerate diffusion coefficient
#              - the drift function μ(t,x) is bounded and continuous in (t,x), uniformly 1/2
#              Hölder in t and uniformly Lipschitz in x, for some constant L > 0

import numpy as np

#We introduce a random discrete time grid with β > 0 a fixed positive constant,
#(τ_i)i>0 be a sequence of i.i.d. E(β)-exponential random variables.

def RandomTimeGrid(Beta, T):
    # Initialise the random time grid
    arrT = [0]
    sumTau = np.random.exponential(1/Beta)
    # get Nt := max{k : Tk < t}
    while sumTau < T:
        arrT.append(sumTau)
        sumTau += np.random.exponential(1/Beta)

    N_T=len(arrT)-1
    arrT.append(T)

    return arrT, N_T

'''
def Unbiased_Simulation_Markovian_Case(funcG, arrX0, funcMu, arrSigma, Beta, T, nDim):
    # Get a random discrete time grid
    arrTimeGrid, N_T = RandomTimeGrid(Beta, T)

    # Compute (DeltaT_k)k≥0
    arrDeltaT = np.diff(arrTimeGrid)
    
    # Initialize array to store X_hat values
    arrX_hat = np.zeros((N_T + 2, nDim))

    # Set initial value (of dimension d)
    arrX_hat[0] = arrX0

    # Simulate the Delta of the d-dimensional Brownian motion W
    arrDeltaW = np.zeros((N_T+1, nDim))
    for i in range(0, N_T + 1):
        arrDeltaW[i] = np.random.normal(loc=0.0, scale=arrDeltaT[i], size=nDim)

    # Euler scheme loop
    for k in range(N_T+1):
        MuValue_k = funcMu(arrTimeGrid[k], arrX_hat[k])

        # Euler scheme formula
        arrX_hat[k + 1] = arrX_hat[k] + arrDeltaT[k] * MuValue_k + arrSigma @ arrDeltaW[k]

    if N_T > 0 :
        # Initialize the products of the W^1_k of the estimator
        prodW1 = 1

        if nDim > 1:
            arrSigma_transpose_inv = np.linalg.inv(arrSigma.transpose())
        else:
            arrSigma_transpose_inv = 1/arrSigma

        # W^1_k loop
        for k in range(N_T+1):
            prodW1 *= ((funcMu(arrTimeGrid[k+1], arrX_hat[k+1]) - funcMu(arrTimeGrid[k], arrX_hat[k]))*arrSigma_transpose_inv*arrDeltaW[k])/arrDeltaT[k]

        Psi_hat = np.exp(Beta*T)*(funcG(arrX_hat[-1]) - funcG(arrX_hat[N_T]))*Beta**(-N_T)*prodW1

    else :
        Psi_hat = np.exp(Beta*T)*funcG(arrX_hat[-1])

    return Psi_hat
'''

def Unbiased_Simulation_Markovian_Case_1D(funcG, X0, funcMu, Sigma, Beta, T):
    # Get a random discrete time grid
    arrTimeGrid, N_T = RandomTimeGrid(Beta, T)

    # Compute (DeltaT_k)k≥0
    arrDeltaT = np.diff(arrTimeGrid)

    # Initialize array to store X_hat values
    X_hat = np.zeros(N_T + 2)

    # Set initial value
    X_hat[0] = X0

    # Simulate the Delta of the d-dimensional Brownian motion W
    arrDeltaW = np.zeros(N_T+1)
    for i in range(0, N_T + 1):
        arrDeltaW[i] = np.random.normal(loc=0.0, scale=arrDeltaT[i])

    # Euler scheme loop
    for k in range(N_T+1):
        MuValue_k = funcMu(arrTimeGrid[k], X_hat[k])

        # Euler scheme formula
        X_hat[k + 1] = X_hat[k] + arrDeltaT[k] * MuValue_k + Sigma * arrDeltaW[k]

    if N_T > 0 :
        # Initialize the products of the W^1_k of the estimator
        prodW1 = 1

        Sigma_transpose_inv = 1/Sigma

        # W^1_k loop
        for k in range(N_T+1):
            prodW1 *= ((funcMu(arrTimeGrid[k+1],X_hat[k+1]) - funcMu(arrTimeGrid[k], X_hat[k]))*Sigma_transpose_inv*arrDeltaW[k])/arrDeltaT[k]

        Psi_hat = np.exp(Beta*T)*(funcG(X_hat[-1]) - funcG(X_hat[N_T]))*Beta**(-N_T)*prodW1

    else :
        Psi_hat = np.exp(Beta*T)*funcG(X_hat[-1])

    return Psi_hat



def MC_estimator(funcG, arrX0, funcMu, arrSigma, Beta, T, nDim, nSamples):

    psi_hats=np.zeros(nSamples)

    for i in range(nSamples):
        psi_hats[i] = Unbiased_Simulation_Markovian_Case_1D(funcG, arrX0, funcMu, arrSigma, Beta, T)

    p=np.mean(psi_hats)
    s=np.std(psi_hats)

    return p,[p-1.96*s/np.sqrt(nSamples),p+1.96*s/np.sqrt(nSamples)], s/np.sqrt(nSamples) #test,confidence interval,error
