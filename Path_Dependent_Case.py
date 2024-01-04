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
    lprodW1 = np.ones(numIter)

    # We respect the index notations
    for k in range(numIter-1): ########### TO BE CHECKED
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
            Sigma_transpose_inv = 1/Sigma
            # W^1_k loop
            for j in range(1, N_t1t2 + 1):
                prodW1[k] *= ((funcMu_k(k, lX_ti, arr_t1t2[j], arrX_tilde[j], numIter) - funcMu_k(k, lX_ti, arr_t1t2[j-1], arrX_tilde[j-1], numIter)) * arrSigma_transpose_inv*arrDeltaW[j])/arrDeltaT[j]


############################################
            Psi_tilde_1 = np.exp(Beta*T)*(funcG(arrX_hat[-1]) - funcG(arrX_hat[N_T]))*Beta**(-N_T)*prodW1
            Psi_tilde_2 = np.exp(Beta*T)*(funcG(arrX_hat[-1]) - funcG(arrX_hat[N_T]))*Beta**(-N_T)*prodW1

        else:
            Psi_tilde_1 = np.exp(Beta*(lTimeIntervals[k+1] - lTimeIntervals[k]))*funcG(arrX_hat[-1])


    return Psi_hat




############ RECURSIVE IMPLEMENTATION ###########

def prodWeight(T, x_k, x_k_minus_1, W_k):
    # Initialize the products of the W^1_k of the estimator
    Sigma_transpose_inv = 1/Sigma
    # W^1_k loop
    for j in range(1, N_t1t2 + 1):
        prodW1[k] *= ((funcMu_k(k, lX_ti, arr_t1t2[j], arrX_tilde[j], numIter) - funcMu_k(k, lX_ti, arr_t1t2[j-1], arrX_tilde[j-1], numIter)) * arrSigma_transpose_inv*arrDeltaW[j])/arrDeltaT[j]
    return prodW1

#lTimeIntervals = (t1,..., tn)
def Psi_US_1D_Recursive(k, Xk, X0, funcG, funcMu, Sigma, Beta, lTimeIntervals):

    if k == len(lTimeIntervals):
        return funcG(Xk) # to be checked
    elif k == 0:
        Xk = [X0]

    Nk_tilde, arr_t1t2, arrDelta_t1t2, arrDeltaW_t1t2 = BrownianMotionSimulation_Interval(Beta, lTimeIntervals[k], lTimeIntervals[k+1])

    # Initialize array to store X_tilde values
    arrX_tilde = np.zeros(N_t1t2 + 2)

    # Set initial value
    arrX_tilde[0] = X0 ################### ATTENTION MUST CHANGE EACH TIMES

    # local Euler scheme loop on [tk, tk+1]
    for j in range(N_t1t2+1):
        # Euler scheme formula
        arrX_tilde[j+1] = arrX_tilde[j] + arrDelta_t1t2[j] * funcMu_k(k, lX_ti, arr_t1t2[j], arrX_tilde[j], numIter) + Sigma * arrDeltaW_t1t2[j]

    lX_ti.append(arrX_tilde[-1])







    Xk_tilde = np.zeros()
    Xk_tilde[0] = X[-1]

    numIter = len(lTimeIntervals)

    lX_ti = []
    lprodW1 = np.ones(numIter)

    t2 = lTimeIntervals[k]
    t1 = lTimeIntervals[k-1]

    X.append()

    return np.exp(Beta*(t2-t1))*(Psi_US_1D_Recursive(k, Xk, funcG, funcMu, Sigma, Beta, lTimeIntervals) - Psi_US_1D_Recursive(k, Xk_0, funcG, funcMu, Sigma, Beta, lTimeIntervals))*Beta**(-)*prodWeight(k,Xk_tilde)





    # We respect the index notations
    for k in range(numIter-1): ########### TO BE CHECKED
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
            Sigma_transpose_inv = 1/Sigma
            # W^1_k loop
            for j in range(1, N_t1t2 + 1):
                prodW1[k] *= ((funcMu_k(k, lX_ti, arr_t1t2[j], arrX_tilde[j], numIter) - funcMu_k(k, lX_ti, arr_t1t2[j-1], arrX_tilde[j-1], numIter)) * arrSigma_transpose_inv*arrDeltaW[j])/arrDeltaT[j]


            Psi_tilde_1 = np.exp(Beta*T)*(funcG(arrX_hat[-1]) - funcG(arrX_hat[N_T]))*Beta**(-N_T)*prodW1
            Psi_tilde_2 = np.exp(Beta*T)*(funcG(arrX_hat[-1]) - funcG(arrX_hat[N_T]))*Beta**(-N_T)*prodW1

        else:
            Psi_tilde_1 = np.exp(Beta*(lTimeIntervals[k+1] - lTimeIntervals[k]))*funcG(arrX_hat[-1])


    return Psi_hat



############ RECURSIVE IMPLEMENTATION ###########


def MC_estimator(funcG, arrX0, funcMu, arrSigma, Beta, T, nDim, nSamples):

    psi_hats=np.zeros(nSamples)

    for i in range(nSamples):
        #psi_hats[i] = Unbiased_Simulation_Path_Dependent_Case_1D((funcG, X0, funcMu, Sigma, Beta, lTimeIntervals))
        psi_hats[i] = Unbiased_Simulation_Path_Dependent_Case_1D_Recursive(0, funcG, X0, funcMu, Sigma, Beta, lTimeIntervals)

    p=np.mean(psi_hats)
    s=np.std(psi_hats)

    return p,[p-1.96*s/np.sqrt(nSamples),p+1.96*s/np.sqrt(nSamples)], s/np.sqrt(nSamples) #test,confidence interval,error

