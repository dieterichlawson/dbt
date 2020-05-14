import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import util

def window_for_points(Xs):
  minx, miny = np.min(Xs, axis=0)
  maxx, maxy = np.max(Xs, axis=0)
  xlength = maxx - minx
  xbounds = (minx - 0.25*xlength, maxx + 0.25*xlength)
  ylength = maxy - miny
  ybounds = (miny - 0.25*ylength, maxy + 0.25*ylength)
  if xbounds[1]-xbounds[0] > ybounds[1] - ybounds[0]:
    extra_length = ((xbounds[1]-xbounds[0]) - (ybounds[1] - ybounds[0]))/2.
    ybounds = (ybounds[0]-extra_length, ybounds[1]+extra_length)
  else:
    extra_length = ((ybounds[1]-ybounds[0]) - (xbounds[1] - xbounds[0]))/2.
    xbounds = (xbounds[0]-extra_length, xbounds[1]+extra_length)
  return xbounds, ybounds


def plot_objective_and_iterates(Xs, iterates, C, k, objective=None, num_pts=100):

  num_points = Xs.shape[0]
  has_nan_iterates = np.any(np.isnan(iterates))
  it_inds = np.logical_not(np.any(np.isnan(iterates), axis=1))
  iterates = iterates[it_inds,:]

  all_Xs = np.concatenate([Xs, iterates], axis=0)
  xbounds, ybounds = window_for_points(all_Xs)

  if objective is not None:
    x = np.arange(xbounds[0], stop=xbounds[1], step=(xbounds[1]-xbounds[0])/float(num_pts))[:num_pts]
    y = np.arange(ybounds[0], stop=ybounds[1], step=(ybounds[1]-ybounds[0])/float(num_pts))[:num_pts]

    X, Y = np.meshgrid(x, y)
    XY = np.reshape(np.stack([X,Y], axis=-1), [num_pts**2, 2])
    objective = jax.vmap(objective)
    density = np.reshape(objective(XY), [num_pts, num_pts])

    if np.any(np.isnan(density)) or np.any(np.isinf(density)):
      print("Density plot has nans")
  else:
    density = np.zeros([num_pts, num_pts])

  fig = plt.figure()
  canvas = FigureCanvas(fig)
  ax1 = fig.gca()
  ax1.imshow(density, extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]], origin="lower")

  for i in range(num_points):
    if i == k:
      ax1.plot(Xs[i,0], Xs[i,1], 'o', color='white')
    else:
      ax1.plot(Xs[i,0], Xs[i,1], 'o', color='fuchsia')
    ax1.annotate(i, (Xs[i,0], Xs[i,1]), color='white')

  for i in range(iterates.shape[0]):
    if i == iterates.shape[0]-1 and has_nan_iterates:
      ax1.plot(iterates[i,0], iterates[i,1], '*', color='crimson')
    else:
      ax1.plot(iterates[i,0], iterates[i,1], 'o', color='crimson')

    if i > 0:
      plt.plot([iterates[i-1,0], iterates[i,0]], [iterates[i-1,1], iterates[i,1]], color='crimson')

  square_C = scipy.spatial.distance.squareform(C)
  for i in range(num_points):
    for j in range(i, num_points):
      if square_C[i,j] < 1.:
        ax1.plot([Xs[i,0], Xs[j,0]], [Xs[i,1], Xs[j,1]], color='fuchsia')

  crimson_line = mlines.Line2D([], [], color='crimson', marker='o',
                              markersize=8, label='Newton iterates')
  fuchsia_line = mlines.Line2D([], [], color='fuchsia', marker='o',
                               markersize=8, label='Posterior means')
  ax1.legend(handles=[crimson_line, fuchsia_line], facecolor='white')

  fig.tight_layout(pad=0)

  # To remove the huge white borders
  ax1.margins(0)
  canvas.draw()
  image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return image_from_plot[np.newaxis,:,:,:]


def plot_posterior(Xs, i, C, L, num_pts=100):
  num_points = Xs.shape[0]
  xb, yb= window_for_points(Xs)
  fig = plt.figure(figsize=(8,4))
  canvas = FigureCanvas(fig)
  ax = [fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)]
  ax[0].set_xlim(left=xb[0], right=xb[1])
  ax[0].set_ylim(bottom=yb[0], top=yb[1])
  ax[1].set_xlim(left=xb[0], right=xb[1])
  ax[1].set_ylim(bottom=yb[0], top=yb[1])

  x = np.arange(xb[0], stop=xb[1], step=(xb[1]-xb[0])/float(num_pts))[:num_pts]
  y = np.arange(yb[0], stop=yb[1], step=(yb[1]-yb[0])/float(num_pts))[:num_pts]
  X, Y = np.meshgrid(x, y)
  XY = np.reshape(np.stack([X,Y], axis=-1), [num_pts**2, 2])

  log_density = jax.vmap(lambda x: util.mv_normal_logpdf(x, Xs[i], L))
  log_density_im = np.reshape(log_density(XY), [num_pts, num_pts])
  density_im = np.exp(log_density_im)

  ax[0].imshow(log_density_im, extent=[xb[0], xb[1], yb[0], yb[1]], origin="lower")
  ax[0].contour(X, Y, log_density_im, extent=[xb[0], xb[1], yb[0], yb[1]], origin="lower")
  ax[0].set_title("Log Posterior over point %d" % i)

  ax[1].imshow(density_im, extent=[xb[0], xb[1], yb[0], yb[1]], origin="lower")
  ax[1].contour(X, Y, density_im, extent=[xb[0], xb[1], yb[0], yb[1]], origin="lower")
  ax[1].set_title("Posterior over point %d" % i)

  for i in range(num_points):
    ax[0].plot(Xs[i,0], Xs[i,1], 'o', color='fuchsia')
    ax[0].annotate(i, (Xs[i,0], Xs[i,1]), color='white')
    ax[1].plot(Xs[i,0], Xs[i,1], 'o', color='fuchsia')
    ax[1].annotate(i, (Xs[i,0], Xs[i,1]), color='white')

  square_C = scipy.spatial.distance.squareform(C)
  for i in range(num_points):
    for j in range(i, num_points):
      if square_C[i,j] < 1.:
        ax[0].plot([Xs[i,0], Xs[j,0]], [Xs[i,1], Xs[j,1]], color='fuchsia')
        ax[1].plot([Xs[i,0], Xs[j,0]], [Xs[i,1], Xs[j,1]], color='fuchsia')

  fig.tight_layout(pad=0)
  canvas.draw()
  image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return image_from_plot[np.newaxis,:,:,:]

def confidence_ellipses(center, cov, ax, **kwargs):
    val, vec = np.linalg.eigh(cov)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))
    for i in range(1, 4):
      percent = scipy.stats.norm.cdf(float(i)) - scipy.stats.norm.cdf(-float(i))
      s = scipy.stats.chi2.ppf(percent, df=2)
      sqrt_lambda_1, sqrt_lambda_2 = np.sqrt(val)
      ellipse = Ellipse(center,
          width=2*sqrt_lambda_1*np.sqrt(s),
          height=2*sqrt_lambda_2*np.sqrt(s),
          angle=rotation,
          fill=False,
          linewidth=.75,
          **kwargs)
      ax.add_patch(ellipse)

def plot_truth_and_posterior(gt_Xs, post_Xs, C, covs, num_pts=100):
  num_points = gt_Xs.shape[0]
  centered_gt_Xs = gt_Xs - np.mean(gt_Xs, axis=0)
  centered_post_Xs = post_Xs - np.mean(post_Xs, axis=0)
  R, scale = scipy.linalg.orthogonal_procrustes(centered_post_Xs, centered_gt_Xs)
  post_Xs = np.dot(centered_post_Xs, R)
  covs = np.matmul(R.T[np.newaxis,:,:], np.matmul(covs, R[np.newaxis,:,:]))
  gt_Xs = centered_gt_Xs

  gtb = window_for_points(gt_Xs)
  pb = window_for_points(post_Xs)

  fig = plt.figure(figsize=(8,4))
  canvas = FigureCanvas(fig)
  ax = [fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)]
  ax[0].set_xlim(left=gtb[0][0], right=gtb[0][1])
  ax[0].set_ylim(bottom=gtb[1][0], top=gtb[1][1])
  ax[1].set_xlim(left=pb[0][0], right=pb[0][1])
  ax[1].set_ylim(bottom=pb[1][0], top=pb[1][1])

  x = np.arange(pb[0][0], stop=pb[0][1], step=(pb[0][1]-pb[0][0])/float(num_pts))[:num_pts]
  y = np.arange(pb[1][0], stop=pb[1][1], step=(pb[1][1]-pb[1][0])/float(num_pts))[:num_pts]
  X, Y = np.meshgrid(x, y)
  XY = np.reshape(np.stack([X,Y], axis=-1), [num_pts**2, 2])

  #batched_log_density = jax.jit(jax.vmap(util.mv_normal_logpdf, in_axes=(None, 0, 0)))
  #sum_density = lambda x: jnp.exp(jscipy.special.logsumexp(batched_log_density(x, post_Xs, Ls)))
  #batched_sum_density = jax.jit(jax.vmap(sum_density, in_axes=0))
  #density = np.reshape(batched_sum_density(XY), [num_pts, num_pts])
  #ax[1].imshow(density, extent=[pb[0][0], pb[0][1], pb[1][0], pb[1][1]], origin="lower")
  #ax[1].contour(X, Y, density, extent=[pb[0][0], pb[0][1], pb[1][0], pb[1][1]], origin="lower")

  for i in range(num_points):
    ax[0].plot(gt_Xs[i,0], gt_Xs[i,1], '.', color='darkorange')
    ax[0].annotate(i, (gt_Xs[i,0], gt_Xs[i,1]), color='white')
    ax[1].plot(post_Xs[i,0], post_Xs[i,1], '.', color='darkorange')
    ax[1].annotate(i, (post_Xs[i,0], post_Xs[i,1]), color='white')
    confidence_ellipses(post_Xs[i], covs[i], ax[1], color='deepskyblue')

  square_C = scipy.spatial.distance.squareform(C)
  for i in range(num_points):
    for j in range(i, num_points):
      if square_C[i,j] < 1.:
        ax[0].plot([gt_Xs[i,0], gt_Xs[j,0]], [gt_Xs[i,1], gt_Xs[j,1]], color='darkorange')
        ax[1].plot([post_Xs[i,0], post_Xs[j,0]], [post_Xs[i,1], post_Xs[j,1]], color='fuchsia')

  ax[0].set_facecolor('midnightblue')
  ax[1].set_facecolor('midnightblue')
  ax[0].set_title("Ground truth")
  ax[1].set_title("Posterior")

  fig.tight_layout(pad=0)
  canvas.draw()
  image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return image_from_plot[np.newaxis,:,:,:]
