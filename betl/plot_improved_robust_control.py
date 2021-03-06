import numpy as np
import pickle
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
sns.set_context("paper")

colors = sns.color_palette("muted")
color_bounds = colors[0]
color_detected = colors[0]
color_outlier = colors[3]
color_cost = colors[7]

color_cost_true_post = colors[2]
color_cost_true_prior = colors[4]


with open('data/improved_data_for_plots.pickle', 'rb') as handle:
    data_for_plots = pickle.load(handle)

switches_actual = data_for_plots['switches_actual']
detected = data_for_plots['detected']
bound_list = data_for_plots['bound_list']
cost_list = data_for_plots['cost_list']
cost_ = data_for_plots['cost_']


data = pd.read_pickle('data/improved_dataframe.pickle')

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
fig = plt.figure(figsize=(x, y*0.9))

cost_plot = sns.scatterplot(x="t", y="cost", data=data, marker='.', alpha=0.3, edgecolor="none", color=color_cost, s=5, rasterized=True)
cost_plot_red = sns.scatterplot(x="t", y="cost_above_thresh", data=data, marker='X', edgecolor=color_outlier, color=color_outlier,
                            ax=cost_plot, s=25, rasterized=True)

# cost_plot.set_rasterized(True)
# cost_plot_red.set_rasterized(True)

for time in switches_actual:
    switch = cost_plot.axvline(x=time, linewidth=1, color='k', linestyle='-.')
# for time in detected:
#     detected_plot = cost_plot.axvline(x=time, linewidth=1, color=color_outlier, linestyle=':')

# For the legend
lc = cost_plot.hlines(y=cost_, xmin=1, xmax=1.,
                      linewidth=3, color=color_bounds, linestyle='-', alpha=1)
import matplotlib.lines as mlines
cost_trigger = mlines.Line2D([], [], color=color_outlier, marker='X', linestyle='None')

for bound in bound_list:

    for low in bound['lower']:

        cost_plot.hlines(y=low, xmin=bound['t_start'], xmax=bound['t_end'],
                    linewidth=3, color=color_bounds, linestyle='-', alpha=2 / 100, rasterized=True)

    for high in bound['upper']:
        cost_plot.hlines(y=high, xmin=bound['t_start'], xmax=bound['t_end'],
                    linewidth=3, color=color_bounds, linestyle='-', alpha=2 / 100, rasterized=True)

    emp_cost = plt.hlines(y=bound['cost_mean'], xmin=bound['t_start_cost'], xmax=bound['t_end'],
                          linewidth=2, color=color_cost_true_post, linestyle=':')

for cost_true in cost_list:
    rob_cost = plt.hlines(y=cost_true['prior_cost_true'], xmin=cost_true['t_start'], xmax=cost_true['t_end'],
                          linewidth=2, color=color_cost_true_prior, linestyle=':')
    # plt.hlines(y=cost_true['post_cost_true'], xmin=cost_true['t_start'], xmax=cost_true['t_end'],
    #            linewidth=1, color=color_cost_true_post, linestyle='--')
    # plt.hlines(y=cost_true['total_cost_true'], xmin=cost_true['t_start'], xmax=cost_true['t_end'],
    #            linewidth=1, color='black', linestyle='--')


cost_plot.legend(handles=[switch, cost_trigger, rob_cost, emp_cost, lc], labels=['$T_i$', 'trigger', '$\Delta J(K_0)$', '$J_{\Delta}(\hat N^*)$', '$p(\kappa$)'])
sns.move_legend(cost_plot, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

cost_plot.set_xlabel('time $k$')
cost_plot.set_ylabel('observed cost $\hat J(k)$')

plt.xlim([0, None])
sns.despine(fig=fig)

# adjust margins
fig.subplots_adjust(bottom=0.2, top=0.75, left=0.14, right=0.99)

# plt.savefig('figures/improve-3d-300dpi.png', dpi=300)
plt.savefig('figures/improve-3d.pgf', format='pgf', dpi=300)
plt.close()
# plt.show()