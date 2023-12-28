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


