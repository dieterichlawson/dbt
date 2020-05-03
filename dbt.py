from __future__ import print_function, division
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
import argparse

import opt
from quadrature import ghq
import util
from tb_logging import TensorboardLogger

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
parser.add_argument('--num_newton_steps', type=int, default=20,
                    help='Number of Newton steps used to find the mode.')
parser.add_argument('--newton_lr', type=float, default=0.1,
                    help='Learning rate to use with Newton\'s method.')

parser.add_argument('--logdir', type=str, default='/tmp/dbt',
                    help="Logging directory for Tensorboard summaries")
                    
def np_log_joint(X, S, D,
                 prior='uniform',
                 prior_var=1.,
                 censorship_temp=10,
                 distance_threshold=.5,
                 distance_var=0.1):
  """Computes the joint log probability for the latent position model using only numpy ops.

  Args:
    X: A 2D numpy array of floats of shape (num_pos, 2), the positions.
    S: A 2D numpy array of 0s and 1s of length num_pos*(num_pos-1)/2. The
      censorship indicators. If an element of S is 1 then the distance between
      the corresponding positions was observed. If it is 0, then it was not
      observed.
    D: A 2D numpy array of floats of length num_pos*(num_pos-1)/2. The distances
      between positions. Distances for censored position pairs are ignored.
    prior: Either 'uniform' or 'gaussian', the form of the position prior.
    prior_var: The variance for the position prior if it is Gaussian.
    censorship_temp: The temperature of the sigmoid that defines the censorship
      distribution, is multiplied by the covariates before they are exponentiated.
    distance_threshold: The distance where the probability of censorship is 50%.
    distance_var: The variance of the conditional distribution of distance given
      the positions and censoring indicators.

  Returns:
    log_p: The log probability of the provided observations.
  """
  if prior == "gaussian":
    prior = onp.sum(scipy.stats.multivariate_normal.logpdf(X, mean=[0.,0.], cov=prior_var))
  elif prior == "uniform":
    prior = 0.
  distances = oscipy.spatial.distance.pdist(X)
  censorship_probs = oscipy.special.expit(censorship_temp*(distances - distance_threshold))
  log_p_C = onp.sum(oscipy.stats.bernoulli.logpmf(C, censorship_probs))
  distance_logprobs = oscipy.stats.norm.logpdf(D, loc=distances, scale=onp.sqrt(distance_var))
  log_p_D = onp.sum(distance_logprobs*(1-C))
  return prior + log_p_C + log_p_D

def log_joint(X, C, D,
              prior='uniform',
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
    prior: Either 'uniform' or 'gaussian', the form of the position prior.
    prior_var: The variance for the position prior if it is Gaussian.
    censorship_temp: The temperature of the sigmoid that defines the censorship
      distribution, is multiplied by the covariates before they are exponentiated.
    distance_threshold: The distance where the probability of censorship is 50%.
    distance_var: The variance of the conditional distribution of distance given
      the positions and censoring indicators.

  Returns:
    log_p: The log probability of the provided observations.
  """
  if prior == "gaussian":
    prior = np.sum(scipy.stats.norm.logpdf(X, mean=[0.,0.], cov=prior_var))
  elif prior == "uniform":
    prior = 0.
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

def expected_log_joint(x, i, C, D, num_points,
                       quad_loc=None,
                       quad_cov=None,
                       prior='uniform',
                       prior_var=1.,
                       censorship_temp=10,
                       distance_threshold=.5,
                       distance_var=0.1,
                       integrator=None):
  """Computes E_{q(x_{not i})}[log p(X,D,C)].
  
  Args:
    x: A 1-D numpy array of floats of shape [2]. The location to evaluate.
    i: The index of the location to evaluate, must be in [0, num_points]
    C: A 2D numpy array of 0s and 1s of length num_pos*(num_pos-1)/2. The
      censorship indicators. If an element of C is 1 then the distance between
      the corresponding positions was censored. If it is 0, then it was observed.
    D: A 2D numpy array of floats of length num_pos*(num_pos-1)/2. The distances
      between positions. Distances for censored position pairs are ignored.
    num_points: The total number of location.
    prior: Either 'uniform' or 'gaussian', the form of the position prior.
    prior_var: The variance for the position prior if it is Gaussian.
    censorship_temp: The temperature of the sigmoid that defines the censorship 
      distribution, is multiplied by the covariates before they are exponentiated.
    distance_threshold: The distance where the probability of censorship is 50%.
    distance_var: The variance of the conditional distribution of distance given
      the positions and censoring indicators.
      
  Returns:
    log_p: The log probability of the provided observations.
  """
  if integrator is None:
    integrator = functools.partial(ghq, degree=5)
  if quad_loc is None:
    quad_loc = np.zeros([(num_points-1)*2])
  if quad_cov is None:
    quad_cov = np.eye((num_points-1)*2)

  def f(cond_x):
    cond_x = cond_x.reshape([num_points-1, 2])
    real_x = np.concatenate((cond_x[:i], x[np.newaxis,:], cond_x[i:]))
    return log_joint(real_x, C, D, 
                     prior=prior, 
                     prior_var=prior_var, 
                     censorship_temp=censorship_temp, 
                     distance_threshold=distance_threshold, 
                     distance_var=distance_var)
  f = jit(f)
  return integrator(f, quad_loc, quad_cov)

def grad_expected_log_joint(x, i, C, D, num_points,
                            quad_loc=None,
                            quad_cov=None,
                            prior='uniform',
                            prior_var=1.,
                            censorship_temp=10,
                            distance_threshold=.5,
                            distance_var=0.1,
                            integrator=None):
  """Computes grad w.r.t. x_i of E_{q(x_{not i})}[log p(X,D,C)].
  
  Args:
    x: A 1-D numpy array of floats of shape [2]. The location to evaluate.
    i: The index of the location to evaluate, must be in [0, num_points]
    C: A 2D numpy array of 0s and 1s of length num_pos*(num_pos-1)/2. The
      censorship indicators. If an element of C is 1 then the distance between
      the corresponding positions was censored. If it is 0, then it was observed.
    D: A 2D numpy array of floats of length num_pos*(num_pos-1)/2. The distances
      between positions. Distances for censored position pairs are ignored.
    num_points: The total number of location.
    prior: Either 'uniform' or 'gaussian', the form of the position prior.
    prior_var: The variance for the position prior if it is Gaussian.
    censorship_temp: The temperature of the sigmoid that defines the censorship 
      distribution, is multiplied by the covariates before they are exponentiated.
    distance_threshold: The distance where the probability of censorship is 50%.
    distance_var: The variance of the conditional distribution of distance given
      the positions and censoring indicators.
      
  Returns:
    log_p: The log probability of the provided observations.
  """
  if integrator is None:
    integrator = functools.partial(ghq, degree=5)

  if quad_loc is None:
    quad_loc = np.zeros([(num_points-1)*2])
  if quad_cov is None:
    quad_cov = np.eye((num_points-1)*2)
  
  def f(x_i, x_other):
    x_other = x_other.reshape([num_points-1, 2])
    real_x = np.concatenate((x_other[:i], x_i[np.newaxis,:], x_other[i:]))
    return log_joint(real_x, C, D, 
                     prior=prior, 
                     prior_var=prior_var, 
                     censorship_temp=censorship_temp, 
                     distance_threshold=distance_threshold, 
                     distance_var=distance_var)

  gradf = jit(grad(f, argnums=0))
  gradf = functools.partial(gradf, x)
  return integrator(gradf, quad_loc, quad_cov)

def hess_expected_log_joint(x, i, C, D, num_points,
                            quad_loc=None,
                            quad_cov=None,
                            prior='uniform',
                            prior_var=1.,
                            censorship_temp=10,
                            distance_threshold=.5,
                            distance_var=0.1,
                            integrator=None):
  """Computes grad^2 w.r.t. x_i of E_{q(x_{not i})}[log p(X,D,C)].
  
  Args:
    x: A 1-D numpy array of floats of shape [2]. The location to evaluate.
    i: The index of the location to evaluate, must be in [0, num_points]
    C: A 2D numpy array of 0s and 1s of length num_pos*(num_pos-1)/2. The
      censorship indicators. If an element of C is 1 then the distance between
      the corresponding positions was censored. If it is 0, then it was observed.
    D: A 2D numpy array of floats of length num_pos*(num_pos-1)/2. The distances
      between positions. Distances for censored position pairs are ignored.
    num_points: The total number of location.
    prior: Either 'uniform' or 'gaussian', the form of the position prior.
    prior_var: The variance for the position prior if it is Gaussian.
    censorship_temp: The temperature of the sigmoid that defines the censorship 
      distribution, is multiplied by the covariates before they are exponentiated.
    distance_threshold: The distance where the probability of censorship is 50%.
    distance_var: The variance of the conditional distribution of distance given
      the positions and censoring indicators.
      
  Returns:
    log_p: The log probability of the provided observations.
  """
  if integrator is None:
    integrator = functools.partial(ghq, degree=5)

  if quad_loc is None:
    quad_loc = np.zeros([(num_points-1)*2])
  if quad_cov is None:
    quad_cov = np.eye((num_points-1)*2)
  
  def f(x_i, x_other):
    x_other = x_other.reshape([num_points-1, 2])
    real_x = np.concatenate((x_other[:i], x_i[np.newaxis,:], x_other[i:]))
    return log_joint(real_x, C, D, 
                     prior=prior, 
                     prior_var=prior_var, 
                     censorship_temp=censorship_temp, 
                     distance_threshold=distance_threshold, 
                     distance_var=distance_var)

  hessf = jit(jax.hessian(f, argnums=0))
  hessf = functools.partial(hessf, x)
  return integrator(hessf, quad_loc, quad_cov)

def dbt_laplace(num_points, C, D,
                init_mus=None,
                prior='uniform',
                prior_var=1.,
                censorship_temp=10,
                distance_threshold=.5,
                distance_var=0.1,
                num_steps=1000,
                num_newton_steps=25,
                newton_lr=0.05):
  # initialize q parameters
  if init_mus is None:
    mus = onp.random.normal(size=(num_points,2))
  covs = np.array([np.eye(2)]*num_points)

  # set up function, grad, and hessian
  def elj(x, i, quad_loc, quad_cov):
    return expected_log_joint(
        x, i, C, D, num_points,
        quad_loc=quad_loc, quad_cov=quad_cov,
        prior=prior, prior_var=prior_var,
        censorship_temp=censorship_temp,
        distance_threshold=distance_threshold,
        distance_var=distance_var,
        integrator=functools.partial(ghq, degree=3))

  def grad_elj(x, i, quad_loc, quad_cov):
    return grad_expected_log_joint(
        x, i, C, D, num_points,
        quad_loc=quad_loc, quad_cov=quad_cov,
        prior=prior, prior_var=prior_var,
        censorship_temp=censorship_temp,
        distance_threshold=distance_threshold,
        distance_var=distance_var,
        integrator=functools.partial(ghq, degree=3))

  def hess_elj(x, i, quad_loc, quad_cov):
    return hess_expected_log_joint(
        x, i, C, D, num_points,
        quad_loc=quad_loc, quad_cov=quad_cov,
        prior=prior, prior_var=prior_var,
        censorship_temp=censorship_temp,
        distance_threshold=distance_threshold,
        distance_var=distance_var,
        integrator=functools.partial(ghq, degree=3))

  for t in range(num_steps):
    for i in range(num_points):
      # Use newton's method to find the mode
      quad_loc = onp.concatenate([mus[:i], mus[i+1:]], axis=0).reshape([(num_points-1)*2])
      quad_cov = scipy.linalg.block_diag(*[covs[j] for j in range(num_points) if j != i])
      xs, fs = opt._newtons_method(lambda x: elj(x, i, quad_loc, quad_cov),
                                   lambda x: grad_elj(x, i, quad_loc, quad_cov),
                                   lambda x: hess_elj(x, i, quad_loc, quad_cov),
                                   newton_lr, num_newton_steps, mus[i])
      # update mu and Sigma
      new_mu_i = xs[-1]
      new_cov_i = -onp.linalg.inv(hess_elj(new_mu_i, i, quad_loc, quad_cov))
      print(new_cov_i)
      print(onp.linalg.eig(new_cov_i)[0])
      mus = onp.concatenate([mus[:i], [new_mu_i], mus[i+1:]], axis=0)
      covs = onp.concatenate([covs[:i], [new_cov_i], covs[i+1:]], axis=0)

  return mus, covs


args = parser.parse_args()

num_points = 3
censorship_temp = 0.
distance_var = 0.01
distance_threshold = 5.
X, C, D = sample(args.num_points,
                 censorship_temp=args.censorship_temp,
                 distance_var=args.distance_variance,
                 distance_threshold=args.distance_threshold)

#plot_problem(X, C, D, censorship_temp=censorship_temp, distance_var=distance_var, distance_threshold=distance_threshold)
logger = TensorboardLogger(args.logdir)

dbt_laplace(args.num_points, C, D,
            censorship_temp=args.censorship_temp,
            distance_var=args.distance_variance,
            distance_threshold=args.distance_threshold,
            num_steps=args.num_steps,
            num_newton_steps=args.num_newton_steps, newton_lr=args.newton_lr)
