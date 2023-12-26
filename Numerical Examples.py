import numpy as np

from Markovian_Case import Unbiased_Simulation_Markovian_Case

# First Example
# V0 := E[sin(X_T)]
# where X is defined by the SDE, for some constant μ0 ∈ R,
# X0 = 0, dXt = μ0 cos(Xt)dt + 0.5dWt


# Parameters:
nDim = 1
X_0 = 0
Mu_0 = 0.2
Beta = 0.5  # Beta constant
Sigma_0 = 0.5
def funcMu (t,x) :
    return np.cos(x)*Mu_0
T = 100

print(Unbiased_Simulation_Markovian_Case(np.sin, X_0, funcMu, Sigma_0, Beta, T, nDim))
