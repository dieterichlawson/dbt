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

def adam(f, grad_f, init_x, lr, num_steps=25, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
  first_moment = np.zeros_like(init_x)
  second_moment = np.zeros_like(init_x)
  x = init_x
  xs = [x]
  fs = [f(x)]
  for t in range(num_steps):
    g = grad_f(x)
    first_moment = beta_1*first_moment + (1 - beta_1)*g
    second_moment = beta_2*second_moment + (1-beta_2)*(g**2)
    true_first_moment = first_moment/(1. - np.power(beta_1, t+1))
    true_second_moment = second_moment/(1. - np.power(beta_2, t+1))
    x = x - lr*true_first_moment/(np.sqrt(true_second_moment) + epsilon)
    xs.append(x)
    fs.append(f(x))
  return np.array(xs), np.array(fs)

