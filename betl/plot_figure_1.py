import numpy as np
import copy
import pandas as pd

from betl.linear_system import DiscreteTimeLinearSystem as LinearSystem


import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARN)

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

# sns.set_context("paper", rc=params)

def plot_1d_result(result):


    settings = result['settings']

    system = LinearSystem(settings['system']['A'], settings['system']['B'], settings['system']['V'])
    K_prior = result['K0']
    # controller = StateFeedbackLaw(K=K_prior)
    # system.controller = controller

    T = settings['T']

    ussm_prior = result['prior_ussm']
    N_s = result['Ns']
    K_s = result['Ks']
    beta_data = result['beta_data']
    synthesis_settings = result['synthesis_settings']

    J_K0_true = result['J_K0_true']
    J_pi_true = result['J_pi_true']

    J_K_opt_true = result['J_K_opt_true']
    J_K0 = result['J_K0']
    G = result['G']
    J_pi = result['J_pi']
    E_J_K0 = result['E_J_K0']
    E_G = result['E_G']
    factor = result['factor']
    E_J_pi = result['E_J_pi']

    print('-----------------------------------------------------------')

    A_prior, B_prior, V_prior = ussm_prior.sample(2000, c=synthesis_settings['confidence_interval'])

    data_list = list()
    for A, B, V in zip(A_prior, B_prior, V_prior):

        system_sample = dict()
        system_sample['A'] = A[0][0]
        system_sample['B'] = B[0][0]
        system_sample['V'] = V[0][0]
        system_sample['dist'] = 'prior'

        data_list.append(system_sample)

    systems_prior = pd.DataFrame.from_dict(data_list)

    # prior_plot = sns.kdeplot(data=systems_prior, x="A", y="B", fill=True)
    # plt.plot(system.A, system.B, 'rx', ms=10)
    # plt.show()

    def plot_1d_ussm(axis, ussm, prior_list, color_prior, color_post, color_true):

        A_post, B_post, V_post = ussm.sample(2000, c=synthesis_settings['confidence_interval'])
        # data_list = list()
        data_list = copy.deepcopy(prior_list)
        for A, B, V in zip(A_post, B_post, V_post):
            system_sample = dict()
            system_sample['A'] = A[0][0]
            system_sample['B'] = B[0][0]
            system_sample['V'] = V[0][0]
            system_sample['dist'] = 'posterior'

            data_list.append(system_sample)

        import pandas as pd
        systems_post = pd.DataFrame.from_dict(data_list)
        prior = systems_post.loc[systems_post['dist'] == 'prior']
        post = systems_post.loc[systems_post['dist'] == 'posterior']

        plot = sns.kdeplot(ax=axis, data=post, x="A", y="B", levels=10, fill=True, color=color_post)
        plot.plot(system.A, system.B, 'x', mew=2.5, ms=8, color=color_true)
        # plt.show()

        return plot

    from matplotlib.gridspec import GridSpec

    # CDC column width is 245pt, for double column plot change to 505pt
    x, y = set_size(245,
                    subplots=(1, 1),  # specify subplot layout for nice scaling
                    fraction=1.)  # scale width/height

    colors = sns.color_palette("deep", 5)
    colors[4] = 'black'
    sns.set_palette(colors)


    # def format_axes(fig):
    #     for i, ax in enumerate(fig.axes):
    #         ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
    #         ax.tick_params(labelbottom=False, labelleft=False)


    fig = plt.figure(figsize=(x, y))

    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1, sharex=ax1)

    ax2.set(ylabel=None)
    ax2.tick_params('y', labelleft=False)

    prior_samples = data_list

    plt_p = plot_1d_ussm(ax1, result['post_ussm'][0]['ussm'], prior_samples, color_prior=colors[0], color_post=colors[0], color_true=colors[4])

    plt_i = plot_1d_ussm(ax2, result['post_ussm'][2]['ussm'], prior_samples, color_prior=colors[0], color_post=colors[2], color_true=colors[4])

    plt_i.plot(system.A - 0.15, system.B + 0.2, 'x', mew=2.5, ms=8, color=colors[3], alpha=0.7)

    # ax2.set(ylabel=None)
    # ax2.set_yticklabels([])

    #### BOTTOM ARROW
    # Create the arrow
    # 1. Get transformation operators for axis and figure
    ax0tr = ax1.transData  # Axis 0 -> Display
    ax1tr = ax2.transData  # Axis 1 -> Display
    figtr = fig.transFigure.inverted()  # Display -> Figure
    # 2. Transform arrow start point from axis 0 to figure coordinates
    ptB = figtr.transform(ax0tr.transform((1.2, 0.8)))
    # 3. Transform arrow end point from axis 1 to figure coordinates
    ptE = figtr.transform(ax1tr.transform((0.9, 0.8)))
    # 4. Create the patch
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
        fc="k", connectionstyle="arc3,rad=0.3", arrowstyle='simple', alpha=0.7,
        mutation_scale=20.
    )

    ax1.text(.95, 0.07, "learn", size=8, ha="center",
             transform=ax1.transAxes, usetex=True)

    ax1.text(.95, 0.73, "trigger", size=8, ha="center",
             transform=ax1.transAxes, usetex=True)
    # 5. Add patch to list of objects to draw onto the figure
    fig.patches.append(arrow)

    # UPPER ARROW
    ax0tr = ax1.transData  # Axis 0 -> Display
    ax1tr = ax2.transData  # Axis 1 -> Display
    figtr = fig.transFigure.inverted()  # Display -> Figure
    # 2. Transform arrow start point from axis 0 to figure coordinates
    ptB2 = figtr.transform(ax1tr.transform((0.9, 1.2)))
    # 3. Transform arrow end point from axis 1 to figure coordinates
    ptE2 = figtr.transform(ax0tr.transform((1.2, 1.2)))
    arrow2 = FancyArrowPatch(
        ptB2, ptE2, transform=fig.transFigure,  # Place arrow in figure coord system
        fc="k", connectionstyle="arc3,rad=0.3", arrowstyle='simple', alpha=0.7,
        mutation_scale=20.
    )
    # 5. Add patch to list of objects to draw onto the figure
    fig.patches.append(arrow2)

    # change ARROW
    ax0tr = ax2.transData  # Axis 0 -> Display
    ax1tr = ax2.transData  # Axis 1 -> Display
    figtr = fig.transFigure.inverted()  # Display -> Figure
    # 2. Transform arrow start point from axis 0 to figure coordinates
    ptB2 = figtr.transform(ax1tr.transform((system.A[0][0], system.B[0][0])))
    # 3. Transform arrow end point from axis 1 to figure coordinates
    ptE2 = figtr.transform(ax0tr.transform((system.A[0][0] - 0.15, system.B[0][0] + 0.2)))
    arrow2 = FancyArrowPatch(
        (system.A[0][0], system.B[0][0]), (system.A[0][0] - 0.15, system.B[0][0] + 0.2), # transform=fig.transFigure,  # Place arrow in figure coord system
        fc=colors[3], connectionstyle="arc3,rad=0.3", arrowstyle='simple', alpha=0.7,
        mutation_scale=12.
    )
    # 5. Add patch to list of objects to draw onto the figure
    ax2.add_patch(arrow2)

    ax2.text(1.1, 1.1, "change", size=9, ha="center", usetex=True)

    ax1.set_title(label=r'Robust control $K_0$', size=9)
    ax2.set_title(label=r'Learned control $K_N$', size=9)

    ax1.set_xlabel("$A$", usetex=True)
    ax1.set_ylabel("$B$", usetex=True)

    ax2.set_xlabel("$A$", usetex=True)

    gs.update(left=0.15, right=0.99, top=0.9, bottom=0.2, wspace=0.1, hspace=0)

    # plt.savefig('1d-example-300dpi.eps', dpi=300)
    # plt.savefig('1d-example-300dpi.pdf', dpi=300)

    # plt.show()
    plt.savefig('figures/algo.pgf', format='pgf')
    plt.close()

if __name__ == "__main__":
    np.random.seed(2)
    import pickle

    with open(f'data/1d-result-data.pickle', 'rb') as file:
        result = pickle.load(file)

    plot_1d_result(result)
