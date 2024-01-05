import numpy as np

# Euler scheme for simulating SDEs over a fixed time grid.
def Euler_Scheme_Gen(arrX0, funcMu, funcSigma, T, nDim, mSteps):
    dt=T/mSteps #time step size
    X=np.zeros((mSteps+1,nDim))
    X[0]=arrX0 #inialize at X0
    time_grid=np.linspace(0, T, mSteps+1) #give the grid (t0,...,tm=T)

    for i in range(mSteps):
        dW = np.sqrt(dt)*np.random.randn(nDim)
        X[i+1] = X[i] + funcMu(time_grid[i], X[i])*dt + funcSigma(time_grid[i], X[i]) @ dW
    return X


# Monte Carlo simulation for Markovian payoff
def MC_EulerScheme_Markovian(funcG, arrX0, funcMu, funcSigma, T, nDim, mSteps, nSamples):
    g_hats = np.zeros(nSamples)

    for i in range(nSamples):
        g_hats[i] = funcG(Euler_Scheme_Gen(arrX0, funcMu, funcSigma,T, nDim,mSteps)[-1])

    p = np.mean(g_hats)
    s = np.std(g_hats)

    return p, [p - 1.96 * s / np.sqrt(nSamples), p + 1.96 * s / np.sqrt(nSamples)], s / np.sqrt(nSamples)  # test,confidence interval,error

# Monte Carlo simulation for path-dependent payoff
def MC_EulerScheme_Pathdep_Example(arrX0, funcMu, funcSigma, T, nDim, mSteps, nSamples,K):
    g_hats = np.zeros(nSamples)

    for i in range(nSamples):
        path = Euler_Scheme_Gen(arrX0, funcMu, funcSigma, T, nDim, mSteps)
        g_hats[i] = np.maximum(np.mean( np.exp(path[1:]) ) - K, 0)

    p = np.mean(g_hats)
    s = np.std(g_hats)

    return p, [p - 1.96 * s / np.sqrt(nSamples), p + 1.96 * s / np.sqrt(nSamples)], s / np.sqrt(nSamples)  # test,confidence interval,error

