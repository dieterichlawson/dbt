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
from plot import plot_objective_and_iterates, plot_truth_and_posterior, plot_posterior

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default="laplace",
                    choices=["laplace", "mf"],
                    help='Algorithm to run.')
parser.add_argument('--problem_type', type=str, default="random",
                    choices=["random", "lattice"],
                    help='Whether to sample the problem randomly or use a lattice.')
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

parser.add_argument('--make_posterior_plots', type=bool, default=True,
                    help="If true, log posterior plots to Tensorboard (expensive).")
parser.add_argument('--make_optimization_plots', type=bool, default=False,
                    help="If true, log optimization plots to Tensorboard (expensive).")
parser.add_argument('--logdir', type=str, default='/tmp/dbt',
                    help="Logging directory for Tensorboard summaries")

@jit
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
  prior = np.sum(scipy.stats.norm.logpdf(X, loc=0., scale=np.sqrt(prior_var)))
  distances = util.jax_l2_pdist(X)
  censorship_probs = scipy.special.expit(censorship_temp*(distances - distance_threshold))
  log_p_C = np.sum(scipy.stats.bernoulli.logpmf(C, censorship_probs))
  distance_logprobs = scipy.stats.norm.logpdf(D, loc=distances, scale=np.sqrt(distance_var))
  log_p_D = np.sum(distance_logprobs*(1-C))
  return prior + log_p_C + log_p_D


def sample(N,
           X=None,
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
  if X is None:
    X = onp.random.multivariate_normal(mean=[0.,0.], cov=prior_var*np.identity(2), size=[N])
  distances = oscipy.spatial.distance.pdist(X)
  censorship_probs = oscipy.special.expit(censorship_temp*(distances - distance_threshold))
  C = onp.random.binomial(1, censorship_probs)
  uncensored_D = onp.random.normal(loc=distances, scale=onp.sqrt(distance_var))
  D = uncensored_D*(1-C)
  return X, C, D


def lattice(size=10, lims=(-1,1)):
  ticks = onp.linspace(lims[0], lims[1], num=size)
  Xs, Ys = onp.meshgrid(ticks, ticks)
  X = onp.reshape(onp.stack([Xs, Ys], axis=-1),[-1,2])
  num_points = X.shape[0]

  true_D = oscipy.spatial.distance.pdist(X)

  C = []
  for i in range(int(num_points*(num_points-1)/2)):
    m, n = util.flat_to_rc(i, num_points)
    if (n % size != 0 and (m == (n + 1) or m == (n-1))) or (m == (n + size) or m == (n - size)):
      C.append(0)
    else:
      C.append(1)
  C = onp.array(C)
  return X, C, true_D*(1-C)

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
  covs = np.array([np.eye(2)]*num_points)

  def elbo(X, q_locs, q_scales):
    log_p = log_joint(X, C, D,
            prior_var=prior_var,
            censorship_temp=censorship_temp,
            distance_threshold=distance_threshold,
            distance_var=distance_var)
    log_q = util.batched_mv_normal_logpdf(X, q_locs, q_scales)
    return log_p - np.sum(log_q)

  # set up function, grad, and hessian
  def lj(x, i, cond_x):
    real_x = np.concatenate((cond_x[:i], x[np.newaxis,:], cond_x[i:]))
    return -log_joint(real_x, C, D,
                      prior_var=prior_var,
                      censorship_temp=censorship_temp,
                      distance_threshold=distance_threshold,
                      distance_var=distance_var)

  batched_lj = jit(vmap(lj, in_axes=(None, None, 0)), static_argnums=1)

  grad_lj = jax.grad(lj, argnums=0)
  batched_grad_lj = jit(vmap(grad_lj, in_axes=(None, None, 0)), static_argnums=1)

  hess_lj = jax.jacfwd(grad_lj, argnums=0)
  batched_hess_lj = jit(vmap(hess_lj, in_axes=(None, None, 0)), static_argnums=1)

  for t in range(num_outer_steps):

    if plot and logger is not None:
      logger.log_images("truth_vs_posterior_mean", plot_truth_and_posterior(X, mus, C, covs), t)

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
        h = expected_hess_lj(new_mu_i)
        new_cov_i = np.linalg.inv(h)
        new_L_i = np.linalg.cholesky(new_cov_i)
        if np.any(np.isnan(new_L_i)) or np.any(np.diag(new_L_i) < 1e-4):
          print("  Hessian at new x is not positive definite, discarding.")
          new_mus = mus
        else:
          new_mus = np.concatenate([mus[:i], new_mu_i[np.newaxis,:], mus[i+1:]], axis=0)
          Ls = np.concatenate([Ls[:i], new_L_i[np.newaxis,:], Ls[i+1:]], axis=0)
          covs = np.concatenate([covs[:i], new_cov_i[np.newaxis,:], covs[i+1:]], axis=0)

      if plot and logger is not None:
        logger.log_images("objective/%d" % i,
                plot_objective_and_iterates(mus, xs, C, i, objective=lambda x:-expected_lj(x)),
                t)

      mus = new_mus

  return mus, Ls

def dbt_mf(num_points, C, D, opt_method,
           init_mus=None,
           prior_var=1.,
           censorship_temp=10,
           distance_threshold=.5,
           distance_var=0.1,
           num_outer_steps=100,
           lr=0.05,
           logger=None):
  # initialize q parameters
  if init_mus is None:
    mus = onp.random.normal(size=(num_points,2)).astype(onp.float32)
  log_diag = np.zeros((num_points,2), dtype=np.float32)
  off_diag = np.zeros((num_points,1), dtype=np.float32)
  params = np.concatenate([mus, log_diag, off_diag], axis=1)

  # function to unpack the collapsed parameters into usable arrays
  def unpack_params(params):
    params = params.reshape((-1, params.shape[-1]))
    locs = params[...,0:2]
    diag = np.exp(params[...,2:4])
    off_diag = params[...,4]
    Ls = np.stack([diag[...,0], np.zeros_like(off_diag), off_diag, diag[...,1]], axis=1)
    return locs, Ls.reshape((-1,2,2))

  # Make quadrature points and weights
  std_pts_and_wts = quadrature.std_ghq_2d_separable(num_points-1)
  std_pts_and_wts_1d = quadrature.std_ghq_2d_separable(1)
  std_pts_and_wts_1d = (std_pts_and_wts_1d[0].squeeze(), std_pts_and_wts_1d[1])
  
  def lj(xi, i, other_x):
    real_x = util.insert(other_x, xi, i)
    return log_joint(real_x, C, D,
                     prior_var=prior_var,
                     censorship_temp=censorship_temp,
                     distance_threshold=distance_threshold,
                     distance_var=distance_var)

  batched_lj = vmap(lj, in_axes=(None, None, 0))

  def kl(q_i_params, i, q_not_i_params):
    loc_i, L_i = unpack_params(q_i_params)
    loc_i = loc_i.squeeze(0)
    L_i = L_i.squeeze(0)
    cov_i = np.dot(L_i, L_i.T)
    loc_not_i, L_not_i = unpack_params(q_not_i_params)

    # Set up mapping from [degree^2, num_points-1, 2] random Gaussian noise to
    # [degree^2] samples from num_points-1 different 2d Gaussians.
    q_not_i_map = lambda pts: quadrature.batched_transform_points(pts, loc_not_i, L_not_i)
    # [degree^2, 2] -> [degree^2 , 2]
    q_i_map = lambda pts: quadrature.transform_points(pts, loc_i, L_i)

    batched_lj_reparam = lambda xi, i, eps: batched_lj(xi, i, q_not_i_map(eps))

    def log_target(xi):
      return quadrature.integrate_std(
              lambda eps: batched_lj_reparam(xi, i, eps),
              std_pts_and_wts)

    batch_log_target = vmap(log_target)
    batch_log_target_reparam = lambda eps: batch_log_target(q_i_map(eps))
    return (-util.jax_mv_normal_entropy(cov_i)
            - quadrature.integrate_std(batch_log_target_reparam, std_pts_and_wts_1d))

  grad_kl = grad(kl, argnums=0)

  def inner_opt(q_i_params, i, q_not_i_params):
    return opt_method(lambda x: kl(x, i, q_not_i_params),
                      lambda x: grad_kl(x, i, q_not_i_params),
                      lambda x: 0.,
                      q_i_params)

  inner_opt = jit(inner_opt)

  for t in range(num_outer_steps):
    print('Global step %d' % (t+1))
    if args.make_posterior_plots and logger is not None:
      mus, Ls = unpack_params(params)
      covs = np.matmul(Ls, Ls.transpose(axes=(0,2,1)))
      logger.log_images("truth_vs_posterior_mean", plot_truth_and_posterior(X, mus, C, covs), t)

    for i in range(num_points):
      print('  Inner step %d' % (i+1))
      q_i_params = params[i]
      q_not_i_params = np.concatenate([params[:i], params[i+1:]], axis=0)

      new_params = inner_opt(q_i_params, i, q_not_i_params)

      # update mu and Sigma
      if np.any(np.isnan(new_params)):
        print("  New X is nan, discarding")
        print(new_params)
      else:
        params = np.concatenate([params[:i], new_params[np.newaxis,:], params[i+1:,:]], axis=0)

      if args.make_optimization_plots and logger is not None:
        logger.log_images("objective/%d" % i,
                plot_objective_and_iterates(params[:,0:2], new_params[:,0:2], C, i),
                t)
  return mus, Ls

args = parser.parse_args()

if args.problem_type == "random":
  X, C, D = sample(args.num_points,
                   censorship_temp=args.censorship_temp,
                   distance_var=args.distance_variance,
                   distance_threshold=args.distance_threshold)
else:
  sqrt = int(np.sqrt(args.num_points))
  assert sqrt**2 == args.num_points, "num_points must be a square to use lattice."
  X, C, D = lattice(int(np.sqrt(args.num_points)))

logger = TensorboardLogger(args.logdir)

if args.algo == "laplace":
  algo = dbt_laplace
else:
  algo = dbt_mf

algo(args.num_points, C, D,
     opt_method=opt.get_opt_method(args.opt_method, args.num_inner_opt_steps, lr=args.lr),
     censorship_temp=args.censorship_temp,
     distance_var=args.distance_variance,
     distance_threshold=args.distance_threshold,
     num_outer_steps=args.num_steps,
     lr=args.lr,
     logger=logger)
