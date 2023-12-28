import numpy as np

import Markovian_Case
import Euler_Scheme

# First Example
# V0 := E[(ST − K)+]

'''
# Parameters:
nDim = 1
X_0 = 0
Beta = 0.05 # Beta constant
Sigma_0 = np.array([0.5])
M = 4 # large
K = 1 # strike
T = 1
N = 10**5
EulerScheme_mSteps = 10

def funcMu (t,x):
    return 0.1 * (np.sqrt(np.minimum(M, np.exp(x))) - 1.0) - 0.125
def funcSigma (t,x) :
    return Sigma_0
def funcG (x) :
    return np.maximum(0, np.exp(x) - K)


#print(Markovian_Case.Unbiased_Simulation_Markovian_Case(funcG, X_0, funcMu, Sigma_0, Beta, T, nDim))
print(Markovian_Case.MC_estimator(funcG, X_0, funcMu, Sigma_0, Beta, T, nDim, N))
#print(Euler_Scheme.MC_EulerScheme(funcG, X_0, funcMu, funcSigma, T, nDim, EulerScheme_mSteps, N))
'''



# TEST 1D ALONE

# Parameters:
nDim = 1
X_0 = 0
Beta = 0.1 # Beta constant
Sigma_0 = 0.5
M = 4 # large
K = 1 # strike
T = 1
N = 10**5
EulerScheme_mSteps = 10

def funcMu (t,x):
    return 0.1 * (np.sqrt(np.min([M, np.exp(x)])) - 1.0) - 0.125
def funcSigma (t,x) :
    return Sigma_0
def funcG (x) :
    return np.max([0, np.exp(x) - K])


#print(Markovian_Case.Unbiased_Simulation_Markovian_Case(funcG, X_0, funcMu, Sigma_0, Beta, T, nDim))
print(Markovian_Case.MC_estimator(funcG, X_0, funcMu, Sigma_0, Beta, T, nDim, N))
#print(Euler_Scheme.MC_EulerScheme(funcG, X_0, funcMu, funcSigma, T, nDim, EulerScheme_mSteps, N))
