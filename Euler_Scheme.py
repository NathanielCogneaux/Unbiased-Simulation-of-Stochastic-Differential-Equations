# In this section we define the different Euler discretization Schemes
# we are using for the different examples

import numpy as np

# Euler scheme for simulating SDEs over a fixed time grid.
def Euler_Scheme(X0, funcMu, funcSigma, T, nDim, mSteps):
    #time step size
    dt = T / mSteps
    X = np.zeros((mSteps+1,nDim))

    # the Euler scheme at X0
    X[0] = X0
    # Get the grid (t0,...,tm=T) with steps dt
    time_grid = np.linspace(0, T, mSteps + 1)

    # Euler scheme loop
    for i in range(mSteps):

        # Brownian motion simulation
        dW = np.sqrt(dt) * np.random.randn(nDim)

        # Euler scheme formula
        X[i+1] = X[i] + funcMu(time_grid[i], X[i])*dt + funcSigma(time_grid[i], X[i]) @ dW

    return X


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
    
def EulerScheme_Numex3(m, N, sigma, K, x, T):

    h = T / m  # Time step size
    X = x + np.zeros(N)  # Initial values

    for j in range(m):
        W = np.random.randn(N) * np.sqrt(h)
        X += (2 * sigma * W) / (1 + X**2)  # SDE dynamics

    payoff = np.maximum(X - K, 0)  # Payoff function
    mean_payoff = np.mean(payoff)
    std_payoff = np.std(payoff)

    return mean_payoff,std_payoff / np.sqrt(N)
"""


# We now provide a Monte Carlo estimation of Euler Scheme in the Markovian Case
def MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, funcSigma, T, nDim, mSteps, nSamples):

    g_hats = np.zeros(nSamples)

    for i in range(nSamples):
        g_hats[i] = funcG(Euler_Scheme(X0, funcMu, funcSigma, T, nDim, mSteps)[-1])

    p = np.mean(g_hats)
    s = np.std(g_hats)

    #test, statistical confidence interval, statistical error
    return p, [p - 1.96 * s / np.sqrt(nSamples), p + 1.96 * s/np.sqrt(nSamples)], s / np.sqrt(nSamples)


# We now provide a Monte Carlo estimation of Euler Scheme for a path-dependent payoff
def MC_estimator_EulerScheme_Pathdep(funcG, X0, funcMu, funcSigma, T, nDim, mSteps, nSamples):

    g_hats = np.zeros(nSamples)

    for i in range(nSamples):
        g_hats[i] = funcG(Euler_Scheme(X0, funcMu, funcSigma, T, nDim, mSteps)[1:])

    p = np.mean(g_hats)
    s = np.std(g_hats)

    #test, statistical confidence interval, statistical error
    return p, [p - 1.96 * s / np.sqrt(nSamples), p + 1.96 * s / np.sqrt(nSamples)], s / np.sqrt(nSamples)
