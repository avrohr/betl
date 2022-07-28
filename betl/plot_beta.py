import numpy as np
import pandas as pd
import pickle
import seaborn as sns

with open('data/1d-beta_data_for_plots.pickle', 'rb') as handle:
    data_for_plots = pickle.load(handle)

x_data = data_for_plots['x_data']
y_data = data_for_plots['y_data']
beta_plot = data_for_plots['beta_plot']
n_beta = data_for_plots['n_beta']
GNs = data_for_plots['GNs']
GN = data_for_plots['GN']
G_param = data_for_plots['G_param']
beta_parameter = data_for_plots['beta_parameter']


results = pd.read_pickle('data/1d-beta_dataframe.pickle')

import matplotlib as mpl

# Use the pgf backend (must be set before pyplot imported)
mpl.use('pgf')

from utils.postprocessing_utils import initialize_plot, set_size
c, params = initialize_plot('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)

import seaborn as sns
sns.set_style("whitegrid")

import matplotlib.pyplot as plt
plt.rcParams.update(params)

# CDC column width is 245pt, for double column plot change to 505pt
x, y = set_size(245,
                subplots=(1, 1),  # specify subplot layout for nice scaling
                fraction=1.)  # scale width/height
fig = plt.figure(figsize=(x, y*.85))


col = sns.color_palette("deep")
grey = col[7]
blue = col[0]
plt.scatter(x_data, y_data, label='samples', marker='+', s=2, color=grey)
plt.plot(beta_plot, G_param, label='fitted', linewidth=2, color=blue)
plt.scatter(n_beta[1:], GN[1:], label='mean', color='black', marker='x')
# splt.fill_between(n_beta, GN-np.std(GNs, axis=1), GN+np.std(GNs, axis=1), facecolor='#386B94', alpha=0.5)

axes = plt.gca()
axes.legend(loc='lower right')

# axes.set_xlabel(r'number of excitation steps $N$')
# axes.set_ylabel(r'$\beta$')

plt.ylabel(r'$\beta$')
plt.xlabel(r'number of excitation steps $N$')

plt.legend(loc="lower right",
           borderaxespad=0.2, ncol=1, frameon=True, labelspacing=0.2, )

# adjust margins
fig.subplots_adjust(bottom=0.24, top=0.95, left=0.17, right=0.98)

# plt.show()

plt.savefig('figures/beta-example.pgf', format='pgf')
plt.close()