from __future__ import print_function, division
import jax
import jax.lax
import jax.random
import jax.numpy as np
import jax.scipy as scipy
from jax import jit, grad, vmap

import numpy as onp
import scipy as oscipy

import matplotlib.pyplot as plt

import functools
from functools import partial
import itertools
import argparse

import opt
import quadrature
from quadrature import ghq
import util
from tb_logging import TensorboardLogger
from plot import plot_objective_and_iterates, plot_truth_and_posterior

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=5, 
                    help='Number of points in sample problem.')
parser.add_argument('--censorship_temp', type=float, default=1.,
                    help='Censorship temperature.')
parser.add_argument('--distance_threshold', type=float, default=2.,
                    help='Distance threshold at which the probability of censorship is %50.')
parser.add_argument('--distance_variance', type=float, default=0.1,
                    help='Variance of the noise added to distance observations.')

parser.add_argument('--num_steps', type=int, default=100,
                    help='Number of steps to run the VI algorithm for.')
parser.add_argument('--num_inner_opt_steps', type=int, default=20,
                    help='Number of optimization steps used to find the mode.')
parser.add_argument('--opt_method', type=str, default=opt.ADAM,
                     choices=opt.OPT_METHODS,
                     help='Optimization method.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Learning rate to use with the optimization method.')

parser.add_argument('--make_plots', type=bool, default=False,
                    help="If true, log plots to Tensorboard (expensive).")
parser.add_argument('--logdir', type=str, default='/tmp/dbt',
                    help="Logging directory for Tensorboard summaries")

@partial(jit, static_argnums=(1,2))
def log_joint(X, C, D,
              prior_var=1.,
              censorship_temp=10,
              distance_threshold=.5,
              distance_var=0.1):
  """Computes the joint log probability for the latent position model.

  Args:
    X: A 2D numpy array of floats of shape (num_pos, 2), the positions.
    C: A 2D numpy array of 0s and 1s of length num_pos*(num_pos-1)/2. The
      censorship indicators. If an element of C is 1 then the distance between
      the corresponding positions was censored. If it is 0, then it was observed.
    D: A 2D numpy array of floats of length num_pos*(num_pos-1)/2. The distances
      between positions. Distances for censored position pairs are ignored.
    prior_var: The variance for the position prior if it is Gaussian.
    censorship_temp: The temperature of the sigmoid that defines the censorship
      distribution, is multiplied by the covariates before they are exponentiated.
    distance_threshold: The distance where the probability of censorship is 50%.
    distance_var: The variance of the conditional distribution of distance given
      the positions and censoring indicators.

  Returns:
    log_p: The log probability of the provided observations.
  """
  prior = np.sum(scipy.stats.multivariate_normal.logpdf(X, mean=np.array([0.,0.]), cov=prior_var))
  distances = util.jax_l2_pdist(X)
  censorship_probs = scipy.special.expit(censorship_temp*(distances - distance_threshold))
  log_p_S = np.sum(util.jax_bernoulli_logpmf(C, censorship_probs))
  distance_logprobs = scipy.stats.norm.logpdf(D, loc=distances, scale=np.sqrt(distance_var))
  log_p_D = np.sum(distance_logprobs*(1-C))
  return prior + log_p_S + log_p_D

def sample(N,
           prior='uniform',
           prior_var=1.,
           censorship_temp=10,
           distance_threshold=.5,
           distance_var=0.1):
  """Samples from the latent position model.

  Args:
    N: The number of latent positions.
    prior: Either 'uniform' or 'gaussian', the form of the position prior.
    prior_var: The variance for the position prior if it is Gaussian.
    censorship_temp: The temperature of the sigmoid that defines the censorship
      distribution, is multiplied by the covariates before they are exponentiated.
    distance_threshold: The distance where the probability of censorship is 50%.
    distance_var: The variance of the conditional distribution of distance given
      the positions and censoring indicators.

  Returns:
    X: The positions.
    C: The censorship indicators, a 2D numpy array of 0s and 1s
      of length num_pos*(num_pos-1)/2. Is 0 if a distance was observed and 1 otherwise.
    D: The pairwise distance matrix, flattened. Distances that were censored
      are 0.
  """
  if prior == "gaussian":
    X = onp.random.multivariate_normal(mean=[0.,0.], cov=prior_var*np.identity(2), size=[N])
  elif prior == "uniform":
    X = onp.random.uniform(low=-1., high=1., size=[N,2])
  distances = oscipy.spatial.distance.pdist(X)
  censorship_probs = oscipy.special.expit(censorship_temp*(distances - distance_threshold))
  C = onp.random.binomial(1, censorship_probs)
  uncensored_D = onp.random.normal(loc=distances, scale=onp.sqrt(distance_var))
  D = uncensored_D*(1-C)
  return X, C, D

def dbt_laplace(num_points, C, D, opt_method,
                init_mus=None,
                prior_var=1.,
                censorship_temp=10,
                distance_threshold=.5,
                distance_var=0.1,
                num_outer_steps=100,
                lr=0.05,
                logger=None,
                plot=False):
  # initialize q parameters
  if init_mus is None:
    mus = onp.random.normal(size=(num_points,2)).astype(onp.float32)

  Ls = np.array([np.eye(2)]*num_points)

  # set up function, grad, and hessian
  def lj(x, i, cond_x):
    real_x = np.concatenate((cond_x[:i], x[np.newaxis,:], cond_x[i:]))
    return -log_joint(real_x, C, D, 
                     prior_var=prior_var, 
                     censorship_temp=censorship_temp, 
                     distance_threshold=distance_threshold, 
                     distance_var=distance_var)

  def elbo(X, q_locs, q_scales):
    log_p = log_joint(X, C, D,
            prior_var=prior_var,
            censorship_temp=censorship_temp,
            distance_threshold=distance_threshold,
            distance_var=distance_var)
    log_q = util.batched_mv_normal_logpdf(X, q_locs, q_scales)
    return log_p - np.sum(log_q)

  batched_lj = jit(vmap(lj, in_axes=(None, None, 0)), static_argnums=(1,2))

  grad_lj = jax.grad(lj, argnums=0)
  batched_grad_lj = jit(vmap(grad_lj, in_axes=(None, None, 0)), static_argnums=(1,2))

  hess_lj = jax.hessian(lj, argnums=0)
  batched_hess_lj = jit(vmap(hess_lj, in_axes=(None, None, 0)), static_argnums=(1,2))

  for t in range(num_outer_steps):

    if plot and logger is not None:
      logger.log_images("truth_vs_posterior_mean", plot_truth_and_posterior(X, mus, C), t)

    print('Global step %d' % (t+1))

    pts, weights = quadrature.gauss_hermite_points_and_weights(
            mus, Ls, degree=10)
    step_elbo = jax.jit(jax.vmap(lambda X: elbo(X, mus, Ls), in_axes=0))
    elbo_val = quadrature.integrate(step_elbo, pts, weights)
    print("Elbo: %0.4f" % elbo_val)

    if logger is not None:
      logger.log_scalar("elbo", float(elbo_val), t)

    for i in range(num_points):
      print('  Inner step %d' % (i+1))
      quad_locs = np.concatenate([mus[:i], mus[i+1:]], axis=0)
      quad_scales = np.concatenate([Ls[:i], Ls[i+1:]], axis=0)
      pts, weights = quadrature.gauss_hermite_points_and_weights(
              quad_locs, quad_scales, degree=3)

      def expected_lj(x):
        return quadrature.integrate(lambda cond_x: batched_lj(x, i, cond_x), pts, weights)

      def expected_grad_lj(x):
        return quadrature.integrate(lambda cond_x: batched_grad_lj(x, i, cond_x), pts, weights)

      def expected_hess_lj(x):
        return quadrature.integrate(lambda cond_x: batched_hess_lj(x, i, cond_x), pts, weights)

      xs = opt_method(expected_lj, expected_grad_lj, expected_hess_lj, mus[i])
                
      # update mu and Sigma
      if np.any(np.isnan(xs[-1])):
        print("  New X is nan, discarding")
        new_mus = mus
      else:
        new_mu_i = xs[-1]
        h = -expected_hess_lj(new_mu_i)
        new_cov_i = - np.linalg.inv(h)
        new_L_i = np.linalg.cholesky(new_cov_i)
        new_mus = np.concatenate([mus[:i], new_mu_i[np.newaxis,:], mus[i+1:]], axis=0)
        Ls = np.concatenate([Ls[:i], new_L_i[np.newaxis,:], Ls[i+1:]], axis=0)

      if plot and logger is not None:
        logger.log_images("objective/%d" % i,
                plot_objective_and_iterates(mus, xs, C, i, lambda x:-expected_lj(x)),
                t)

      mus = new_mus

  return mus, Ls

args = parser.parse_args()

X, C, D = sample(args.num_points,
                 censorship_temp=args.censorship_temp,
                 distance_var=args.distance_variance,
                 distance_threshold=args.distance_threshold,
                 prior='gaussian')

logger = TensorboardLogger(args.logdir)

dbt_laplace(args.num_points, C, D,
            opt_method=opt.get_opt_method(args.opt_method, args.num_inner_opt_steps, lr=args.lr),
            censorship_temp=args.censorship_temp,
            distance_var=args.distance_variance,
            distance_threshold=args.distance_threshold,
            num_outer_steps=args.num_steps,
            lr=args.lr,
            logger=logger,
            plot=args.make_plots)
