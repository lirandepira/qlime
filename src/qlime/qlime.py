# Temporary copy from the GitHub repo.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.linear_model import LogisticRegression

from src.qlime.model import qnn

def explain(local_idx, X_train, optimized, model, eps = 0.45, local_samples = 100, local_region = 0.025, n_samples = 25):

  lines = np.zeros([n_samples, 100])

  x = X_train

  x_min, x_max = x[:, 0].min() - 0.0101, x[:, 0].max() + 0.0101
  y_min, y_max = x[:, 1].min() - 0.0101, x[:, 1].max() + 0.0101

  x1, x2 = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

  # Generate lines for plotting by sampling new data points around local points
  for idn in range(n_samples):
    new_X_train = X_train[local_idx,:]+local_region*np.random.randn(local_samples,2)
    new_Y_train = np.array([round(model(new_X_train[y])) for y in range(local_samples)])

    lin = LogisticRegression(fit_intercept = True, C=1e5)
    lin.fit(new_X_train, new_Y_train)

    w = lin.coef_[0]
    m = -w[0] / w[1]
    b = (0.5 - lin.intercept_[0]) / w[1]

    lines[idn] = m*np.linspace(x_min,x_max,100)+b

  Z = []
  for idx1, idx2 in zip(x1.ravel(),x2.ravel()):
    Z.append(round(qnn(np.array([idx1, idx2]), optimized)))

#  return Z

#def explain_plot(local_idx, X_train, model, eps=0.45, local_samples=100, local_region=0.025, n_samples=25):
  #Z= explain(local_idx,X_train,model,eps,local_samples,local_region,n_samples)
  # Create a custom colormap
  custom_colors = ['#F27200', '#004D80']
  custom_cmap = colors.ListedColormap(custom_colors)

  Z = np.array(Z).reshape(x1.shape)
  plt.pcolormesh(x1, x2, Z, alpha=0.4, cmap=custom_cmap)

  plt.plot(x[local_idx,0], x[local_idx,1], marker="o", markersize=12, markeredgecolor="k",
  markerfacecolor="yellow")

  sorted_data = np.sort(lines,0)
  up = sorted_data[int((0.5+eps)*100)*n_samples//100,:]
  down = sorted_data[int((0.5-eps)*100)*n_samples//100,:]
  plt.fill_between(np.linspace(x_min,x_max,100), down, up, color = 'b', alpha=0.25)

  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")

  plt.xlim(x_min, x_max - 0.001)
  plt.ylim(y_min, y_max)
  plt.xticks(())
  plt.yticks(());

  # Add the legend
  elements = ['Marker', 'Local region of indecision']
  plt.legend(elements, loc='lower right')

#optimized= qlime.optimizer.optimize(Xtrain, Ytrain)
#explain(local_idx, xtrain, optimized, model)



