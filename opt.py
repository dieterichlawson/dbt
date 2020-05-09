import jax
import jax.numpy as np
from jax import jit


def newtons_method(f, grad_f, hess_f, init_x, alpha=0.5, beta=0.5, num_steps=25):
  x = init_x
  xs = [x]
  ddir = lambda x: np.linalg.solve(hess_f(x), -grad_f(x))
  ddir = jit(ddir)
  for t in range(num_steps):
    descent_dir = ddir(x)
    lambda_sq = np.dot(grad_f(x).T, descent_dir)
    step_size = 1.
    while f(x + step_size*descent_dir) > f(x) + alpha*step_size*lambda_sq:
      step_size = beta*step_size
    x = x + step_size*descent_dir
    xs.append(x)
  return np.array(xs)

def gradient_descent(f, grad_f, init_x, lr, num_steps=25):
  x = init_x
  xs = [x]
  for t in range(num_steps):
    x = x - lr*grad_f(x)
    xs.append(x)
  return np.array(xs)

def gradient_descent_with_momentum(f, grad_f, init_x, lr, num_steps=25, alpha=0.9):
  x = init_x
  xs = [x]
  prev_ddir = np.zeros_like(init_x)
  for t in range(num_steps):
    g = grad_f(x)
    ddir = alpha*prev_ddir - lr*g
    x = x + ddir
    xs.append(x)
    prev_ddir = ddir
  return np.array(xs)

def adam(f, grad_f, init_x, lr, num_steps=25, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
  first_moment = np.zeros_like(init_x)
  second_moment = np.zeros_like(init_x)
  x = init_x
  xs = [x]
  for t in range(num_steps):
    g = grad_f(x)
    first_moment = beta_1*first_moment + (1 - beta_1)*g
    second_moment = beta_2*second_moment + (1-beta_2)*(g**2)
    true_first_moment = first_moment/(1. - np.power(beta_1, t+1))
    true_second_moment = second_moment/(1. - np.power(beta_2, t+1))
    x = x - lr*true_first_moment/(np.sqrt(true_second_moment) + epsilon)
    xs.append(x)
  return np.array(xs)

NEWTONS_METHOD="newtons"
GRADIENT_DESCENT="grad_descent"
GRADIENT_DESCENT_WITH_MOMENTUM="grad_descent_momentum"
ADAM="adam"

OPT_METHODS = [NEWTONS_METHOD, GRADIENT_DESCENT, GRADIENT_DESCENT_WITH_MOMENTUM, ADAM]

OPT_FNS = {NEWTONS_METHOD: newtons_method,
           GRADIENT_DESCENT: gradient_descent,
           GRADIENT_DESCENT_WITH_MOMENTUM: gradient_descent_with_momentum,
           ADAM: adam}

def get_opt_method(opt_method, num_steps, lr=None):
  assert opt_method in OPT_METHODS
  if opt_method == NEWTONS_METHOD:
    def fn(f, grad_f, hess_f, init_x):
      return newtons_method(f, grad_f, hess_f, init_x, num_steps=num_steps)
  elif (opt_method == "grad_descent" or
        opt_method == "grad_descent_momentum" or
        opt_method == "adam"):
    opt_fn = OPT_FNS[opt_method]
    def fn(f, grad_f, unused_hess_f, init_x):
      return opt_fn(f, grad_f, init_x, lr=lr, num_steps=num_steps)
  return fn
