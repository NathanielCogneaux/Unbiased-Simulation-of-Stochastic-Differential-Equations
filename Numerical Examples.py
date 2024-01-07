# In this section we call the different algorithms to provide our numerical tests
# and computation times

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

import Euler_Scheme
import Markovian_Case
import Path_Dependent_Case

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

start_time = time.time()
estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, funcSigma, T, nDim, mSteps, nSamples)
print("Estimator MC_estimator_EulerScheme_Markovian:", estimator)
print("95% Confidence Interval MC_EulerScheme_Markovian:", confidence_interval)
print("Standard Error MC_EulerScheme_Markovian:", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")

start_time = time.time()
estimator, confidence_interval, error = Markovian_Case.MC_estimator(funcG, X0, funcMu, Sigma0, Beta, T, nDim, nSamples)
print("Estimator US_Markovian_Case:", estimator)
print("95% Confidence Interval US_Markovian_Case:", confidence_interval)
print("Standard Error US_Markovian_Case:", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")

start_time = time.time()
estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu, Sigma0, Beta, lTimeIntervals, nSamples)
print("Estimator US_Path_Dependent_Case:", estimator)
print("95% Confidence Interval US_Path_Dependent_Case:", confidence_interval)
print("Standard Error US_Path_Dependent_Case:", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")



Method = []
Mean_value = []
conf_interval = []
statistical_error = []
Computation_time = []
for i in range(4, 9):
    nSamples = 10 ** i ############
    start_time = time.time()
    estimator, confidence_interval, error = Markovian_Case.MC_estimator(funcG, X0, funcMu, Sigma0, Beta, T, nDim, nSamples)
    Computation_time.append(time.time() - start_time)
    Mean_value.append(estimator)
    conf_interval.append(confidence_interval)
    statistical_error.append(error)
    Method.append(f"US (N = 10^{i})")

    start_time = time.time()
    estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, funcSigma, T, nDim, mSteps, nSamples)
    Computation_time.append(time.time() - start_time)
    Mean_value.append(estimator)
    conf_interval.append(confidence_interval)
    statistical_error.append(error)
    Method.append(f"Euler Scheme (N = 10^{i})")

# Round numbers to 9 decimals and make sure confidence intervals fit in the cell
rounded_mean_value = [round(val, 9) for val in Mean_value]
rounded_statistical_error = [round(val, 9) for val in statistical_error]
rounded_computation_time = [round(val, 9) for val in Computation_time]
formatted_conf_interval = [f"[{round(ci[0], 9)}, {round(ci[1], 9)}]" for ci in conf_interval]

# Sample data:
data = {
    'Method': Method,
    'Mean value': Mean_value,
    'Statistical error': statistical_error,
    '95% Confidence Interval': conf_interval,
    'Computation time': Computation_time
}

# Convert data to a Pandas DataFrame
df = pd.DataFrame(data)

# Adjust the figure size (width, height) to accommodate the data
fig, ax = plt.subplots(figsize=(20, 8))  # You may need to adjust these values

# Hide the axes
ax.axis('off')

# Determine column widths - increase width for confidence interval
colWidths = [0.15, 0.1, 0.1, 0.2, 0.1]

# Create the table with adjusted settings
the_table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=["palegreen"] * len(df.columns),
                     colWidths=colWidths)

# Adjust font size
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)  # Adjust the size as needed

# Scale the table to the figure by setting its dimensions
the_table.scale(1, 1.5)  # The second value increases the row heights

# Tight layout for a neat fit
plt.tight_layout()

# Save the table as an image file
plt.savefig('C:/Users/natha/OneDrive/Bureau/MASEF/S1/MC methods FE applied fi/Numerical results/Markovian Numerical results plot.png')





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

start_time = time.time()
estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Pathdep(funcG_PathDep, X0, funcMu, funcSigma, T, nDim, mSteps, nSamples)
print("Estimator MC_estimator_EulerScheme_Pathdep:", estimator)
print("95% Confidence Interval MC_EulerScheme_Pathdep_Example::", confidence_interval)
print("Standard Error MC_EulerScheme_Pathdep_Example::", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")

start_time = time.time()
estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu, Sigma0, Beta, lTimeIntervals, nSamples)
print("Estimator US_Path_Dependent_Case:", estimator)
print("95% Confidence Interval US_Path_Dependent_Case:", confidence_interval)
print("Standard Error US_Path_Dependent_Case:", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")




Method = []
Mean_value = []
conf_interval = []
statistical_error = []
Computation_time = []
for i in range(4, 8):
    nSamples = 10 ** i
    start_time = time.time()
    estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu, Sigma0, Beta, lTimeIntervals, nSamples)
    Computation_time.append(time.time() - start_time)
    Mean_value.append(estimator)
    conf_interval.append(confidence_interval)
    statistical_error.append(error)
    Method.append(f"US (N = 10^{i})")

    start_time = time.time()
    estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Pathdep(funcG_PathDep, X0, funcMu, funcSigma, T, nDim, mSteps, nSamples)
    Computation_time.append(time.time() - start_time)
    Mean_value.append(estimator)
    conf_interval.append(confidence_interval)
    statistical_error.append(error)
    Method.append(f"Euler Scheme (N = 10^{i})")

# Round numbers to 9 decimals and make sure confidence intervals fit in the cell
rounded_mean_value = [round(val, 9) for val in Mean_value]
rounded_statistical_error = [round(val, 9) for val in statistical_error]
rounded_computation_time = [round(val, 9) for val in Computation_time]
formatted_conf_interval = [f"[{round(ci[0], 9)}, {round(ci[1], 9)}]" for ci in conf_interval]

# Sample data:
data = {
    'Method': Method,
    'Mean value': Mean_value,
    'Statistical error': statistical_error,
    '95% Confidence Interval': conf_interval,
    'Computation time': Computation_time
}

# Convert data to a Pandas DataFrame
df = pd.DataFrame(data)

# Adjust the figure size (width, height) to accommodate the data
fig, ax = plt.subplots(figsize=(20, 8))  # You may need to adjust these values

# Hide the axes
ax.axis('off')

# Determine column widths - increase width for confidence interval
colWidths = [0.15, 0.1, 0.1, 0.2, 0.1]

# Create the table with adjusted settings
the_table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=["palegreen"] * len(df.columns),
                     colWidths=colWidths)

# Adjust font size
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)  # Adjust the size as needed

# Scale the table to the figure by setting its dimensions
the_table.scale(1, 1.5)  # The second value increases the row heights

# Tight layout for a neat fit
plt.tight_layout()

# Save the table as an image file
plt.savefig('C:/Users/natha/OneDrive/Bureau/MASEF/S1/MC methods FE applied fi/Numerical results/Path Dependent Numerical results plot.png')
