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




##TEST for V0 in (4.2) (case d = 1)  (expected result : 0.1267 around)

def funcMu(t,x): #func mu in SDE
    return  0.1 * (np.sqrt(np.exp(x)) - 1) - 0.125

def funcSigma(t,x): #func sigma in SDE
    return Sigma_0

def funcG (x): #payoff G for markovian case
    return np.maximum(0, np.exp(x) - K)

# Parameters
arrX0 = np.array([0]) #Initial value
T = 1 #Maturity
nDim = 1 #Dim of process
mSteps = 10 #Number of time steps in Euler Scheme
nSamples = 100000 #Number of sim of MC
K = 1   #Strike
Sigma_0 = np.array([0.5])   #constant coeff in SDE process


# Run Monte Carlo Simulation

estimator, confidence_interval, error = MC_EulerScheme_Pathdep_Example(arrX0, funcMu, funcSigma, T, nDim, mSteps, nSamples, K)
print("Estimator:", estimator)
print("95% Confidence Interval:", confidence_interval)
print("Standard Error:", error)


estimator, confidence_interval, error = MC_EulerScheme_Markovian(funcG,arrX0, funcMu, funcSigma, T, nDim, mSteps, nSamples)
print("Estimator:", estimator)
print("95% Confidence Interval:", confidence_interval)
print("Standard Error:", error)








