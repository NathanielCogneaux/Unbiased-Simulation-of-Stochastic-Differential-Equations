import numpy as np

"""
This Generalized Euler Scheme (working for any dim and any given function in parameters)
is for the case non-path dependant, for the case path dependant, 
just remove [-1] from this line : 

g_hats[i] = funcG(  Euler_Scheme_Gen(arrX0, funcMu, funcSigma,T, nDim,mSteps)[-1] 

"""


def Euler_Scheme_Gen(arrX0, funcMu, funcSigma,T, nDim,mSteps):
    dt=T/mSteps #time step size
    X=np.zeros((mSteps+1,nDim))
    X[0]=arrX0 #inialize at X0
    time_grid=np.linspace(0, T, mSteps+1) #give the grid (t0,...,tm=T)

    for i in range(mSteps):
        dW = np.sqrt(dt)*np.random.randn(nDim)
        X[i+1]=X[i]+funcMu(time_grid[i],X[i])*dt+funcSigma(time_grid[i],X[i])*dW
    return X

def MC_EulerScheme(funcG,arrX0, funcMu, funcSigma,T, nDim,mSteps,nSamples):
    g_hats = np.zeros(nSamples)

    for i in range(nSamples):
        g_hats[i] = funcG(  Euler_Scheme_Gen(arrX0, funcMu, funcSigma,T, nDim,mSteps)[-1] )

    p = np.mean(g_hats)
    s = np.std(g_hats)

    return p, [p - 1.96 * s / np.sqrt(nSamples), p + 1.96 * s / np.sqrt(nSamples)], s / np.sqrt(nSamples)  # test,confidence interval,error


#Testing Example 3 from Numerical example (driftless SDE)
def funcMu(t, x):
    return 0  # Driftless

def funcSigma(t, x):
    sigma = 0.2  # Set the value of sigma as given in the paper
    return 2 * sigma / (1 + x**2)

def funcG(path):
    K = 1  # Set the strike price K as given in the paper
    return max(path[-1] - K, 0)

# Simulation parameters
arrX0 = np.array([1])  # Initial value X0 = 1
Beta = 0.1  # As specified
T = 1  # Total time
nDim = 1
mSteps = 10  # Number of time steps (e.g., 10 for dt = 1/10)
nSamples = 10**5  # Number of simulations

# Monte Carlo simulation
estimator, confidence_interval, error = MC_EulerScheme(funcG, arrX0, funcMu, funcSigma,T, nDim, mSteps, nSamples)
print("Estimator:", estimator)
print("95% Confidence Interval:", confidence_interval)
print("Standard Error:", error)

