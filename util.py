from __future__ import print_function, division

import jax
from jax import jit
import jax.numpy as np
import jax.scipy as scipy

import numpy as onp

from functools import partial

def flat_to_rc(x, n):
  """Converts indices for a flat upper triangular matrix to row, col indices.
  
  Args:
    x (int): An index into a flattened upper triangular matrix. 
    n (int): The number of rows and columns of the matrix.
    
  Returns:
    row, col: The row and col corresponding to the flattened index.
  """
  x = np.array(x, dtype='int32')
  i = n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5)
  j = x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
  if len(x.shape) == 0:
    return int(i), int(j)
  else:
    return np.hstack((i[:,np.newaxis], j[:,np.newaxis]))
  
def rc_to_flat(x, n):
  """Converts row, col indices for an upper triangular matrix to indices into a 
     flattened array.
  
  Args:
    x (int, int): The rows and columns of the indices into the matrix. Rows must
      be less than columns.
    n (int): The number of rows and columns of the matrix.
    
  Returns:
    k: The index into the flattened array.
  """
  assert np.all(x[:,0] < x[:,1])
  return ((n*(n-1))/2) - ((n-x[:,0])*((n-x[:,0])-1))/2 + x[:,1] - x[:,0] - 1

@partial(jit)
def jax_l2_pdist(X):
  """Computes the pairwise distances between points in X.
  
  Args:
    X: A 2d numpy array, the points.
  Returns:
    dm: The pairwise distances between points in X, as a flattened 
      upper-triangular matrix.
  """
  n = X.shape[0]
  diffs = (X[:, None] - X[None, :])[np.triu_indices(n=n, k=1)]
  return np.linalg.norm(diffs, axis=1)

@partial(jit)
def mv_normal_logpdf(X, loc, scale):
  cov = np.dot(scale, scale.T)
  return scipy.stats.multivariate_normal.logpdf(X, loc, cov)

batched_mv_normal_logpdf = jax.jit(jax.vmap(mv_normal_logpdf, in_axes=0))

def jax_mv_normal_entropy(cov):
  k = cov.shape[0]
  eigvals = np.linalg.eigvalsh(cov)
  eps = 1e-5*np.max(abs(eigvals))
  mask = np.greater(abs(eigvals)-eps, 0.)
  log_nonzero_eigvals = np.where(mask, np.log(eigvals), np.zeros_like(eigvals))
  log_det = np.sum(log_nonzero_eigvals)
  return k/2. + (k/2.)*np.log(2*np.pi) + .5*log_det

def insert(m, r, i):
  n = m.shape[0]
  a = np.concatenate([m, r[np.newaxis,:]], axis=0)
  before_inds = np.arange(n+1)*np.less(np.arange(n+1),i)
  after_inds = (np.arange(n+1)-1)*np.greater(np.arange(n+1),i)
  new_ind = np.ones(shape=[n+1], dtype=np.int32)*np.equal(np.arange(n+1),i)*n
  inds = before_inds + after_inds + new_ind
  return a[inds]

def delete(m, i):
  n = m.shape[0]
  before_inds = np.arange(n-1)*np.less(np.arange(n-1),i)
  after_inds = (np.arange(n-1)+1)*np.greater(np.arange(n-1)+1,i)
  inds = before_inds + after_inds
  return m[inds]
