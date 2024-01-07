# In this section we call the different algorithms to provide our numerical tests
# and computation times

import numpy as np

import Markovian_Case
import Path_Dependent_Case
import Euler_Scheme

#import time
'''
def your_algorithm():
    # Your algorithm's code goes here
    pass

start_time = time.time()
your_algorithm()
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
'''



##### TEST for V0 in (4.2) (expected result : 0.205396 around) #####


# Parameters

X0 = 0  # Initial value
T = 1   # Maturity
nDim = 1    # Dim of process
mSteps = 10 # Number of time steps in Euler Scheme
nSamples = 10**5   # Number of simulations of MC

K = 1   # Strike
Sigma0 = 0.5
Beta = 0.1  # Beta constant
M = 4   # M constant

lTimeIntervals = [0, T]

# μ in the provided SDE
def funcMu(t, x):
    return 0.1 * (np.sqrt(np.exp(x)) - 1) - 0.125
# Sigma in the provided SDE
def funcSigma(t, x):
    return [Sigma0]
# Payoff G in the provided example (Call option)
def funcG(x):
    return np.maximum(0, np.exp(x) - K)
# Payoff G in the provided example (Call option) for the Path Dependent Case
def funcG_PathDep(x):
    return np.maximum(0, np.exp(x[-1]) - K)



# Run The Simulations

print("RESULTS FOR THE MARKOVIAN EXAMPLE 4.2 (expected result : 0.205396 around)")
print(" ")

estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, funcSigma, T, nDim, mSteps, nSamples)
print("Estimator MC_estimator_EulerScheme_Markovian:", estimator)
print("95% Confidence Interval MC_EulerScheme_Markovian:", confidence_interval)
print("Standard Error MC_EulerScheme_Markovian:", error)
print(" ")

estimator, confidence_interval, error = Markovian_Case.MC_estimator(funcG, X0, funcMu, Sigma0, Beta, T, nDim, nSamples)
print("Estimator US_Markovian_Case:", estimator)
print("95% Confidence Interval US_Markovian_Case:", confidence_interval)
print("Standard Error US_Markovian_Case:", error)
print(" ")

estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu, Sigma0, Beta, lTimeIntervals, nSamples)
print("Estimator US_Path_Dependent_Case:", estimator)
print("95% Confidence Interval US_Path_Dependent_Case:", confidence_interval)
print("Standard Error US_Path_Dependent_Case:", error)
print(" ")






##### TEST for V0_tilde in (4.2) (expected result : 0.1267 around) #####


# Parameters:

Beta = 0.05 # Beta constant
lTimeIntervals = [i*T/10 for i in range(0, 11)]

# We adapt the new path dependent payoff to the example
def funcG_PathDep (lX):
    return np.maximum(0, np.sum(np.exp(lX))/len(lX) - K)



# Run The Simulations

print("RESULTS FOR THE PATH DEPENDENT EXAMPLE 4.2 (expected result : 0.1267 around)")
print(" ")

estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Pathdep(funcG_PathDep, X0, funcMu, funcSigma, T, nDim, mSteps, nSamples)
print("Estimator MC_estimator_EulerScheme_Pathdep:", estimator)
print("95% Confidence Interval MC_EulerScheme_Pathdep_Example::", confidence_interval)
print("Standard Error MC_EulerScheme_Pathdep_Example::", error)
print(" ")

estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu, Sigma0, Beta, lTimeIntervals, nSamples)
print("Estimator US_Path_Dependent_Case:", estimator)
print("95% Confidence Interval US_Path_Dependent_Case:", confidence_interval)
print("Standard Error US_Path_Dependent_Case:", error)
print(" ")

