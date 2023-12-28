import numpy as np

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

'''
# Test the function
m, N, sigma, K, x, T = 10, 10**6, 0.2, 1, 1, 1
statistical_error = EulerScheme_Numex3(m, N, sigma, K, x, T)
print("Statistical error:", statistical_error)
'''



"""
import numpy as np

def RandomTimeGrid(Beta, T):
    # Initialise the random time grid
    arrT = [0]
    sumTau = np.random.exponential(1/Beta)
    # get Nt := max{k : Tk < t}
    while sumTau < T:
        arrT.append(sumTau)
        sumTau += np.random.exponential(1/Beta)
    N_T=len(arrT)-1
    arrT.append(T)
    return arrT, N_T
def construct_time_grid_and_NT(T, beta):
    time_grid = [0]
    while True:
        tau = np.random.exponential(1 / beta)
        next_time = time_grid[-1] + tau

        if next_time >= T:
            NT = len(time_grid) - 1  # Calculate N_T before adding T
            time_grid.append(T)
            break

        time_grid.append(next_time)

    return time_grid, NT



# Test with same parameters
T_test, beta_test = 1.0, 0.05


# Using construct_time_grid_and_NT
time_grid_construct, NT_construct = construct_time_grid_and_NT(T_test, beta_test)
print("construct_time_grid_and_NT - Time Grid:", time_grid_construct, "N_T:", NT_construct)

# Using RandomTimeGrid
time_grid_random, NT_random = RandomTimeGrid(beta_test, T_test)
print("RandomTimeGrid - Time Grid:", time_grid_random, "N_T:", NT_random)



"""
