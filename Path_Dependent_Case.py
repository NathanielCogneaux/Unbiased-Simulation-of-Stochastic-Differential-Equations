# The unbiased simulation algorithm
# The path dependent case

import numpy as np

#np.random.seed(123)

def RandomTimeGrid_Interval(Beta, t1, t2):
    # Initialise the random time grid
    arr_t1t2 = [t1]
    sumTau = t1 + np.random.exponential(1/Beta)
    while sumTau < t2:
        arr_t1t2.append(sumTau)
        sumTau += np.random.exponential(1/Beta)

    N_t1t2 = len(arr_t1t2)-1
    arr_t1t2.append(t2)

    return arr_t1t2, N_t1t2

def BrownianMotionSimulation_Interval(Beta, t1, t2):
    # Get a random discrete time grid for the interval
    arr_t1t2, N_t1t2 = RandomTimeGrid_Interval(Beta, t1, t2)

    # Compute (DeltaT_k)k≥0
    arrDelta_t1t2 = np.diff(arr_t1t2)

    # Simulate the Delta of the Brownian motion W
    arrDeltaW_t1t2 = np.zeros(N_t1t2 + 1)
    for i in range(N_t1t2 + 1):
        arrDeltaW_t1t2[i] = np.random.normal(loc=0.0, scale=arrDelta_t1t2[i])

    return N_t1t2, arr_t1t2, arrDelta_t1t2, arrDeltaW_t1t2


def funcMu_k(k, lX_ti, t, x, numIter):
    lX = lX_ti[:k] ########### PAS SUR QUE CE SOIT NECESSAIRE
    for i in range(k, numIter):
        lX.append(x)
    return funcMu(t, lX)

# exemple de funcMu possible
def funcMu(t, lX):
    return(np.sum(lX)/len(lX))


def Unbiased_Simulation_Path_Dependent_Case_1D(funcG, X0, funcMu, Sigma, Beta, lTimeIntervals):

    numIter = len(lTimeIntervals)
    lX_ti = []

    # We respect the index notations
    for k in range(numIter-1):
        N_t1t2, arr_t1t2, arrDelta_t1t2, arrDeltaW_t1t2 = BrownianMotionSimulation_Interval(Beta, lTimeIntervals[k], lTimeIntervals[k+1])

        # Initialize array to store X_tilde values
        arrX_tilde = np.zeros(N_t1t2 + 2)

        # Set initial value
        arrX_tilde[0] = X0 ################### ATTENTION MUST CHANGE EACH TIMES

        # local Euler scheme loop on [tk, tk+1]
        for j in range(N_t1t2+1):
            # Euler scheme formula
            arrX_tilde[j+1] = arrX_tilde[j] + arrDelta_t1t2[j] * funcMu_k(k, lX_ti, arr_t1t2[j], arrX_tilde[j], numIter) + Sigma * arrDeltaW_t1t2[j]

        lX_ti.append(arrX_tilde[-1])

        if N_t1t2 > 0:
            # Initialize the products of the W^1_k of the estimator
            prodW1 = 1
            Sigma_transpose_inv = 1/Sigma
            # W^1_k loop
            for j in range(1, N_t1t2 + 1):
                prodW1 *= ((funcMu_k(k, lX_ti, arr_t1t2[j], arrX_tilde[j], numIter) - funcMu_k(k, lX_ti, arr_t1t2[j-1], arrX_tilde[j-1], numIter))*
                           arrSigma_transpose_inv*arrDeltaW[j])/arrDeltaT[j]


############################################
            Psi_hat = np.exp(Beta*T)*(funcG(arrX_hat[-1]) - funcG(arrX_hat[N_T]))*Beta**(-N_T)*prodW1

        else:
            Psi_hat = np.exp(Beta*T)*funcG(arrX_hat[-1])


    return Psi_hat


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
    for i in range(N_T + 1):
        arrDeltaW[i] = np.random.normal(loc=0.0, scale=arrDeltaT[i])

    # Euler scheme loop
    for k in range(N_T+1):
        # Euler scheme formula
        X_hat[k + 1] = X_hat[k] + arrDeltaT[k] * funcMu(arrTimeGrid[k], X_hat[k]) + Sigma * arrDeltaW[k]

    if N_T > 0:
        # Initialize the products of the W^1_k of the estimator
        prodW1 = 1
        Sigma_transpose_inv = 1/Sigma
        # W^1_k loop
        for k in range(1, N_T+1):
            prodW1 *= ((funcMu(arrTimeGrid[k], X_hat[k]) - funcMu(arrTimeGrid[k-1], X_hat[k-1]))*Sigma_transpose_inv*arrDeltaW[k])/arrDeltaT[k]

        Psi_hat = np.exp(Beta*T)*(funcG(X_hat[-1]) - funcG(X_hat[N_T]))*Beta**(-1*N_T)*prodW1

    else :
        Psi_hat = np.exp(Beta*T)*funcG(X_hat[-1])

    return Psi_hat



def MC_estimator(funcG, arrX0, funcMu, arrSigma, Beta, T, nDim, nSamples):

    psi_hats=np.zeros(nSamples)

    for i in range(nSamples):
        psi_hats[i] = Unbiased_Simulation_Path_Dependent_Case_1D((funcG, X0, funcMu, Sigma, Beta, lTimeIntervals))

    p=np.mean(psi_hats)
    s=np.std(psi_hats)

    return p,[p-1.96*s/np.sqrt(nSamples),p+1.96*s/np.sqrt(nSamples)], s/np.sqrt(nSamples) #test,confidence interval,error

