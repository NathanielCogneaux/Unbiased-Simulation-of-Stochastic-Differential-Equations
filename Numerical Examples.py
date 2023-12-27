import numpy as np

import Markovian_Case


# First Example
# V0 := E[sin(X_T)]
# where X is defined by the SDE, for some constant μ0 ∈ R,
# X0 = 0, dXt = μ0 cos(Xt)dt + 0.5dWt


# Parameters:
nDim = 1
X_0 = 0
Mu_0 = 0.2
Beta = 0.05  # Beta constant
Sigma_0 = 0.5
def funcMu (t,x) :
    return np.cos(x)*Mu_0
T = 0.5
N = 10**5


#print(Markovian_Case.Unbiased_Simulation_Markovian_Case(np.sin, X_0, funcMu, Sigma_0, Beta, T, nDim))
print(Markovian_Case.MC_estimator(np.sin, X_0, funcMu, Sigma_0, Beta, T, nDim, N))

