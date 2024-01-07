# In this section we define the different Euler Scheme discretization
# we are using for the different examples

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

def EulerScheme_Numex3(m, N, sigma, K, x, T):
    """
    Euler Scheme for the SDE: dX_t = 2σ / (1 + X_t^2) dW_t.
    (example 3 of numerical methods)
    Parameters:
    m (int): Number of time steps.
    N (int): Number of simulations.
    sigma (float): Volatility parameter.
    K (float): Strike price.
    x (float): Initial value of the process.
    T (float): Terminal time.

    Returns:
    float: Statistical error (sqrt(Var(estimator) / N)).
    """
    h = T / m  # Time step size
    X = x + np.zeros(N)  # Initial values

    for j in range(m):
        W = np.random.randn(N) * np.sqrt(h)
        X += (2 * sigma * W) / (1 + X**2)  # SDE dynamics

    payoff = np.maximum(X - K, 0)  # Payoff function
    mean_payoff = np.mean(payoff)
    std_payoff = np.std(payoff)

    return mean_payoff,std_payoff / np.sqrt(N)
