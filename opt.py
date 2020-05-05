import jax
import jax.numpy as np
from jax import jit

import numpy as onp

def newtons_method(f, grad_f, hess_f, init_x, alpha=0.5, beta=0.5, num_steps=25):
  x = init_x
  fs = [f(x)]
  xs = [x]
  ddir = lambda x: np.linalg.solve(hess_f(x), -grad_f(x))
  ddir = jit(ddir)
  for t in range(max_num_steps):
    descent_dir = ddir(x)
    lambda_sq = np.dot(grad_f(x).T, descent_dir)
    step_size = 1.
    while f(x + step_size*descent_dir) > f(x) + alpha*step_size*lambda_sq:
      step_size = beta*step_size
    x = x + step_size*descent_dir
    fs.append(f(x))
    xs.append(x)
  return np.array(xs), np.array(fs)

def gradient_descent(f, grad_f, init_x, lr, num_steps=25):
  x = init_x
  fs = [f(x)]
  xs = [x]
  for t in range(num_steps):
    x = x - lr*grad_f(x)
    fs.append(f(x))
    xs.append(x)
  return np.array(xs), np.array(fs)

def gradient_descent_with_momentum(f, grad_f, init_x, lr, num_steps=25, alpha=0.9):
  x = init_x
  fs = [f(x)]
  xs = [x]
  prev_ddir = np.zeros_like(init_x)
  for t in range(num_steps):
    g = grad_f(x)
    ddir = alpha*prev_ddir - lr*g
    x = x + ddir
    fs.append(f(x))
    xs.append(x)
    prev_ddir = ddir
  return np.array(xs), np.array(fs)
