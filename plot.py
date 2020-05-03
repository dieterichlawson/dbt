import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.lines as mlines
import jax

def plot_objective_and_iterates(Xs, iterates, C, k, objective, num_pts=100):

  num_points = Xs.shape[0]

  objective = jax.vmap(objective)
  all_Xs = np.concatenate([Xs, iterates], axis=0)  
  minx, miny = np.min(all_Xs, axis=0)
  maxx, maxy = np.max(all_Xs, axis=0)
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

  x = np.arange(xbounds[0], stop=xbounds[1], step=(xbounds[1]-xbounds[0])/float(num_pts))[:num_pts]
  y = np.arange(ybounds[0], stop=ybounds[1], step=(ybounds[1]-ybounds[0])/float(num_pts))[:num_pts]

  X, Y = np.meshgrid(x, y)
  XY = np.reshape(np.stack([X,Y], axis=-1), [num_pts**2, 2])
  density = np.reshape(objective(XY), [num_pts, num_pts])
  if np.any(np.isnan(density)) or np.any(np.isinf(density)):
    print("Density plot has nans")

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


def plot_truth_and_posterior(gt_Xs, post_Xs, C, num_pts=100):
  num_points = gt_Xs.shape[0]
  all_Xs = np.concatenate([gt_Xs, post_Xs], axis=0)
  minx, miny = np.min(all_Xs, axis=0)
  maxx, maxy = np.max(all_Xs, axis=0)
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

  fig = plt.figure()
  canvas = FigureCanvas(fig)
  ax1 = fig.gca()
  ax1.set_xlim(left=xbounds[0], right=xbounds[1])
  ax1.set_ylim(bottom=ybounds[0], top=ybounds[1])

  for i in range(num_points):
    ax1.plot(gt_Xs[i,0], gt_Xs[i,1], 'o', color='darkorange')
    ax1.annotate(i, (gt_Xs[i,0], gt_Xs[i,1]), color='white')
    ax1.plot(post_Xs[i,0], post_Xs[i,1], 'o', color='fuchsia')
    ax1.annotate(i, (post_Xs[i,0], post_Xs[i,1]), color='white')

  square_C = scipy.spatial.distance.squareform(C)
  for i in range(num_points):
    for j in range(i, num_points):
      if square_C[i,j] < 1.:
        ax1.plot([gt_Xs[i,0], gt_Xs[j,0]], [gt_Xs[i,1], gt_Xs[j,1]], color='darkorange')
        ax1.plot([post_Xs[i,0], post_Xs[j,0]], [post_Xs[i,1], post_Xs[j,1]], color='fuchsia')
  
  ax1.set_facecolor('midnightblue')
  orange_line = mlines.Line2D([], [], color='darkorange', marker='o',
                              markersize=8, label='Truth')
  fuchsia_line = mlines.Line2D([], [], color='fuchsia', marker='o',
                               markersize=8, label='Posterior means')
  ax1.legend(handles=[orange_line, fuchsia_line], facecolor='white')

  fig.tight_layout(pad=0)

  # To remove the huge white borders
  ax1.margins(0)
  canvas.draw()
  image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return image_from_plot[np.newaxis,:,:,:]
