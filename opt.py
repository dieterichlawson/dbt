import jax
import jax.numpy as np
from jax import jit

import numpy as onp

def newtons_method(grad, hess, lr, num_steps, init_x):
  return _newtons_method(lambda x: 1, grad, hess, lr, num_steps, init_x)[0]

def _newtons_method(f, grad_f, hess_f, lr, num_steps, init_x):
  x = init_x
  fs = [f(x)]
  xs = [x]
  ddir = lambda x: np.linalg.solve(hess_f(x), -grad_f(x))
  ddir = jit(ddir)
  for t in range(num_steps):
    descent_dir = ddir(x)
    x = x + lr*descent_dir
    fs.append(f(x))
    xs.append(x)
  return np.array(xs), np.array(fs)

def gradient_descent(grad, lr, num_steps, init_x):
  return _gradient_descent(lambda x: 1, grad, lr, num_steps, init_x)[0]

def _gradient_descent(f, grad_f, lr, num_steps, init_x):
  x = init_x
  fs = [f(x)]
  xs = [x]
  for t in range(num_steps):
    x = x - lr*grad_f(x)
    fs.append(f(x))
    xs.append(x)
  return np.array(xs), np.array(fs)
