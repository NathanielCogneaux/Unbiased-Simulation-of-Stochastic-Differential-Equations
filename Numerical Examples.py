import numpy as np

import Markovian_Case
import Path_Dependent_Case
import Euler_Scheme

'''
# First Example
# V0 := E[(ST − K)+]

# Parameters:
nDim = 1
X_0 = 0
Beta = 0.1 # Beta constant
Sigma_0 = np.array([0.5])
M = 4 # large
K = 1 # strike
T = 1
N = 10**6
EulerScheme_mSteps = 10

def funcMu (t,x):
    return 0.1 * (np.sqrt(np.minimum(M, np.exp(x))) - 1.0) - 0.125
def funcSigma (t,x):
    return Sigma_0
def funcG (x):
    return np.maximum(0, np.exp(x) - K)


#print(Markovian_Case.Unbiased_Simulation_Markovian_Case(funcG, X_0, funcMu, Sigma_0, Beta, T, nDim))
print(Markovian_Case.MC_estimator(funcG, X_0, funcMu, Sigma_0, Beta, T, nDim, N))
print(Euler_Scheme.MC_EulerScheme(funcG, X_0, funcMu, funcSigma, T, nDim, EulerScheme_mSteps, N))
'''


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
    return (0.1 * (np.sqrt(np.min([M, np.exp(x)])) - 1.0) - 0.125)
def funcSigma (t,x) :
    return Sigma_0
def funcG (x) :
    return np.max([0, np.exp(x) - K])


#print(Markovian_Case.Unbiased_Simulation_Markovian_Case(funcG, X_0, funcMu, Sigma_0, Beta, T, nDim))
print(Markovian_Case.MC_estimator(funcG, X_0, funcMu, Sigma_0, Beta, T,N))
#print(Euler_Scheme.MC_EulerScheme(funcG, X_0, funcMu, funcSigma, T, nDim, EulerScheme_mSteps, N))

'''



##TEST for V0 in (4.2) (case d = 1)  (expected result : 0.1267 around)

def funcMu(t,x): #func mu in SDE
    return  0.1 * (np.sqrt(np.exp(x)) - 1) - 0.125
def funcMu2_TEST(t,x): #func mu in SDE
    return  0.1 * (np.sqrt(np.exp(x)) - 1) - 0.125
def funcSigma(t,x): #func sigma in SDE
    return Sigma_0
def funcG (x): #payoff G for markovian case
    return np.maximum(0, np.exp(x) - K)
def funcG2 (x): #payoff G for markovian case
    return np.maximum(0, np.exp(x[-1]) - K)
# Parameters
arrX0 = np.array([0]) #Initial value
X0 = 0
T = 1 #Maturity
nDim = 1 #Dim of process
mSteps = 10 #Number of time steps in Euler Scheme
nSamples = 100000 #Number of sim of MC
N = nSamples
K = 1   #Strike
Sigma_0 = np.array([0.5])   #constant coeff in SDE process
Sigma = 0.5

Beta = 0.1

lTimeIntervals = [0, T]

print("RESULTS FOR THE MARKOVIAN EXAMPLE 4.2 CASE D = 1 (expected result : 0.205396 around)")
print(" ")

estimator, confidence_interval, error = Euler_Scheme.MC_EulerScheme_Markovian(funcG,arrX0, funcMu, funcSigma, T, nDim, mSteps, nSamples)
print("Estimator MC_EulerScheme_Markovian:", estimator)
print("95% Confidence Interval MC_EulerScheme_Markovian:", confidence_interval)
print("Standard Error MC_EulerScheme_Markovian:", error)
print(" ")

estimator, confidence_interval, error = Markovian_Case.MC_estimator(funcG, X0, funcMu, Sigma, Beta, T, nDim, nSamples)
print("Estimator US_Markovian_Case:", estimator)
print("95% Confidence Interval US_Markovian_Case:", confidence_interval)
print("Standard Error US_Markovian_Case:", error)
print(" ")

estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG2, X0, funcMu2_TEST, Sigma, Beta, lTimeIntervals, N)
print("Estimator US_Path_Dependent_Case:", estimator)
print("95% Confidence Interval US_Path_Dependent_Case:", confidence_interval)
print("Standard Error US_Path_Dependent_Case:", error)
print(" ")


##TEST for V0_tilde in (4.2) (case d = 1)  (expected result : 0.1267 around)

print("RESULTS FOR THE PATH DEPENDENT EXAMPLE 4.2 CASE D = 1 (expected result : 0.1267 around)")
print(" ")

# Parameters:
X0 = 0
Beta = 0.05 # Beta constant
Sigma = 0.5
M = 4
K = 1 # strike

N = 10**5
T = 1
lTimeIntervals = [i*T/10 for i in range(1,11)]

def funcMu_PathDep (t,X):
    return 0.1 * (np.sqrt(np.min([M, np.exp(X[-1])])) - 1) - 0.125
def funcMu_PathDep_TEST (t,x):
    return 0.1 * (np.sqrt(np.minimum(M, np.exp(x))) - 1) - 0.125
def funcG_PathDep (lX):
    return np.maximum(0, np.sum(np.exp(lX))/len(lX) - K)


# Run Monte Carlo Simulations

estimator, confidence_interval, error = Euler_Scheme.MC_EulerScheme_Pathdep_Example(arrX0, funcMu, funcSigma, T, nDim, mSteps, nSamples, K)
print("Estimator MC_EulerScheme_Pathdep_Example:", estimator)
print("95% Confidence Interval MC_EulerScheme_Pathdep_Example::", confidence_interval)
print("Standard Error MC_EulerScheme_Pathdep_Example::", error)
print(" ")

estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu_PathDep_TEST, Sigma, Beta, lTimeIntervals, N)
print("Estimator US_Path_Dependent_Case:", estimator)
print("95% Confidence Interval US_Path_Dependent_Case:", confidence_interval)
print("Standard Error US_Path_Dependent_Case:", error)
print(" ")

