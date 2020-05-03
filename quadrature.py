import jax
import jax.lax
import jax.random
import jax.numpy as np
import jax.scipy as scipy
from jax import jit, grad

import numpy as onp
import scipy as oscipy

import matplotlib.pyplot as plt

import functools
import itertools

def ghq(f, loc, cov, degree=5):
  """Computes an estimate of E[f(x)] using Gauss-Hermite quadrature.

  Args:
    f: The function to estimate E[f(x)] for. Must accept a [data_dim] input 
      point and return a scalar or vector. 
    loc: A vector of shape [data_dim], the means of the Normal distributions to 
      integrate against.
    cov: A PSD matrix of shape [data_dim, data_dim], the covariance matrix  of 
      the normal distributions to integrate against.

  Returns:
    The estimate of E[f(x)], a scalar.
  """
  n = loc.shape[0]
  xs_1d, ws_1d = onp.polynomial.hermite.hermgauss(degree)
  xs_nd = onp.array(list(itertools.product(*[xs_1d]*n)))
  ws_nd = onp.prod(onp.array(list(itertools.product(*[ws_1d]*n))), axis=1)
  L = onp.linalg.cholesky(cov)
  g = lambda x: f(np.sqrt(2.)*np.dot(L, x.T).T + loc)
  g = jax.vmap(g, in_axes=0)
  gs = g(xs_nd)
  ws_shape = [gs.shape[0]] + [1]*(gs.ndim-1)
  return np.sum(gs * ws_nd.reshape(ws_shape), axis=0)/np.power(np.pi, (n/2.))

def ghq_separable(f, loc, cov, degree=5):
  """Computes an estimate of E[f(x)] using "separable" Gauss-Hermite quadrature.

  Separable GHQ denotes that f is assumed to be composed of a sum of functions
  each of which depends only on one dimension to be integrated over. This
  allows us to generate points for each dimension independent of the others
  instead of taking the product over all dimensions.

  Args:
    f: The function to estimate E[f(x)] for. Must accept a [data_dim] input 
      point and return a scalar. 
    loc: A vector of shape [data_dim], the means of the Normal distributions to 
      integrate against.
    cov: A PSD matrix of shape [data_dim, data_dim], the covariance matrix  of 
      the normal distributions to integrate against.

  Returns:
    The estimate of E[f(x)], a scalar.
  """
  n = loc.shape[0]
  xs_1d, ws_1d = onp.polynomial.hermite.hermgauss(degree)
  xs_nd = onp.stack([xs_1d]*n, axis=-1)
  ws_nd = ws_1d * n 
  L = onp.linalg.cholesky(cov)
  g = lambda x: f(np.sqrt(2.)*np.dot(L, x.T).T + loc)
  g = jax.vmap(g, in_axes=0)
  gs = g(xs_nd)
  ws_shape = [gs.shape[0]] + [1]*(gs.ndim-1)
  return np.sum(gs * ws_nd.reshape(ws_shape), axis=0)/(n*np.power(np.pi, (1/2.)))

def mc_integrator(f, loc, cov, num_samples):
  """Computes an estimate of E[f(x)] using sampling.

  Args:
    f: The function to estimate E[f(x)] for. Must accept a [data_dim] input 
      point and return a scalar. 
    loc: A vector of shape [data_dim], the means of the Normal distributions to 
      integrate against.
    cov: A PSD matrix of shape [data_dim, data_dim], the covariance matrix  of 
      the normal distributions to integrate against.
    num_samples: The number of samples to use.
  Returns:
    The estimate of E[f(x)], a scalar.
  """
  xs = onp.random.multivariate_normal(mean=loc, cov=cov, size=num_samples)
  f = jax.vmap(f, in_axes=0)
  sample_est = np.mean(f(xs), axis=0)
  return sample_est
