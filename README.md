# Unbiased Simulation of Stochastic Differential Equations

## Overview

This repo implements and explores the algorithm proposed in the paper:
**[Unbiased Simulation of Stochastic Differential Equations](https://arxiv.org/abs/1504.06107)**
by *Pierre Henry-Labordère, Xiaolu Tan, and Nizar Touzi*.

> Developed as part of a **Master’s degree project**, the goal was to analyze an unbiased Monte Carlo estimator for $\mathbb{E}[g(X)]$, where $X$ follows a stochastic differential equation (SDE).

The core idea of the paper is to simulate a **regime-switching diffusion** whose coefficients are updated at random exponential times. This approach avoids the bias introduced by time discretization in traditional schemes like Euler–Maruyama, and uses **Malliavin calculus** techniques (Bismut-Elworthy-Li formula) for automatic differentiation.

## What This Repository Contains

* **PDF Report**:
  The report (`Unbiased Simulation of Stochastic Differential Equations.pdf`) explains the theoretical foundations of the method and reproduces several results from the paper. It includes:

  * A breakdown of the algorithm for both the Markovian and path-dependent cases
  * A comparison with the Euler scheme in terms of error and computation time
  * An attempt to generalize the method to more complex SDEs using transformations like Lamperti's

* **Notebook**:
  The notebook `use_cases.ipynb` showcases practical examples and reproduces numerical results from the paper. It serves as a place to experiment with and understand the implementation.

* **Code Modules**:
  The other Python files that you can find inside `module/` contain the core implementations used by the notebook. These cover different classes of SDEs (Markovian, path-dependent, general) and include basic benchmark methods.

The code is **fully usable**, and the notebook can be run to reproduce our numerical findings.

## Key Insights

* The **Unbiased Simulation (US)** method matches the statistical precision of the Euler scheme, while **eliminating discretization bias**.
* It is **faster** than Euler in some settings (e.g. Markovian SDEs), and only slightly slower in others (e.g. path-dependent cases).
* However, generalizing the approach to arbitrary SDEs introduces strong assumptions and technical challenges.
* Future work may involve hybrid methods (e.g. PDE-based) or adaptive schemes to overcome these limitations.


Install the required dependencies:

All dependencies can be installed at once using:
   ```bash
   pip install -r requirements.txt
   ```

## References

\[1] P. Henry-Labordère, X. Tan, and N. Touzi. *Unbiased simulation of stochastic differential equations*, *Annals of Applied Probability*, 27(6):3305–3341, 2017.
\[2] Julien Claisse. *Monte Carlo and Finite Difference Methods with Applications in Finance*, 2021.