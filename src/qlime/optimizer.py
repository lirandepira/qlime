import numpy as np
from src.qlime.model import qnn

def optimize(X_train, Y_train, a= 2.5, c = 0.25, maxiter = 1e3, shots = int(1e4), alpha = 0.602, gamma = 0.101):
  f = lambda x: objective(x, X_train, Y_train, shots=shots)  # Objective function
  x0 = np.pi * np.random.randn(4)

  xsol = spsa(f, x0, a=a, c=c,
              alpha=alpha, gamma=gamma,
              maxiter=maxiter, verbose=True)
  return xsol

def objective(theta, X, Y, shots=int(1e4)):
  """
  Calculates the objective value for a given set of parameters, input data, and target values.

  Parameters:
      theta (numpy.ndarray): Array of parameters.
      X (numpy.ndarray): Input data.
      Y (numpy.ndarray): Target values.
      shots (int): Number of shots for quantum measurement (default: int(1e4)).

  Returns:
      float: The objective value.

  """
  n_data = X.shape[0]
  to_return = 0

  for idx in range(n_data):
    prediction = qnn(X[idx], theta, shots=shots)
    difference = np.abs(prediction - Y[idx]) ** 2
    to_return += difference

  return to_return / n_data

def spsa(func, x0, a=0.1, c=0.1, alpha=0.602, gamma=0.101, maxiter=100, verbose=False):
  """
  Performs the Simultaneous Perturbation Stochastic Approximation (SPSA) optimization algorithm.

  Parameters:
      func (callable): Objective function to be minimized.
      x0 (numpy.ndarray): Initial guess for the parameters.
      a (float): Perturbation size parameter (default: 0.1).
      c (float): Step size parameter (default: 0.1).
      alpha (float): Exponent for step size decay (default: 0.602).
      gamma (float): Exponent for perturbation size decay (default: 0.101).
      maxiter (int): Maximum number of iterations (default: 100).
      verbose (bool): Whether to print progress messages (default: False).

  Returns:
      numpy.ndarray: Optimized parameters.

  """
  k = 0
  x = x0

  while k < maxiter:
    ak = a / (k + 1) ** alpha  # Step size
    ck = c / (k + 1) ** gamma  # Perturbation size
    delta = 2 * np.random.randint(0, 2, len(x0)) - 1  # Random perturbation (+1 or -1) for each parameter
    xp = x + ck * delta  # Perturbed parameter values (positive direction)
    xm = x - ck * delta  # Perturbed parameter values (negative direction)
    grad = (func(xp) - func(xm)) / (2 * ck) * delta  # Estimated gradient

    x = x - ak * grad

    if verbose and k % int(0.1 * maxiter) == 0:
      fx = func(x)
      print(f"Iteration {k}: f = {fx}")

    k += 1

  return x