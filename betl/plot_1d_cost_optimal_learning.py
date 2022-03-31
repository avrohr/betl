import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from betl.linear_system import DiscreteTimeLinearSystem as LinearSystem
from betl.linear_system import StateFeedbackLaw, ExcitingStateFeedbackLaw
from betl.synthesis.robust_lqr_synth import RLQRSyntheziser

from betl.uncertain_state_space_model import MatrixNormal, UncertainStateSpaceModel
from betl.cost_analysis import LinearQuadraticCostAnalysis, EmpiricalQuadraticCostAnalysis

from betl.excitation_strategy import optimal_signal


import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARN)


def plot_cost_bars(axis, result):

    settings = result['settings']
    J_K_opt_true = result['J_K_opt_true']
    Ns = result['Ns']
    Ks = result['Ks']
    T = settings['T']
    K_prior = result['K0']

    system = LinearSystem(settings['system']['A'], settings['system']['B'], settings['system']['V'])

    data = list()
    cost_oracle = LinearQuadraticCostAnalysis(None, settings['Q'], settings['R'])

    for i in range(len(Ns)):
        data_for_exc = {}
        K = Ks[i]
        N = Ns[i]
        A, B = system.A, system.B
        V = system.V
        J_pi_K0 = cost_oracle.quadratic_cost_with_input(K_prior, A, B, V, settings['excitation_variance'],
                                                        episode_length=1)
        J_K0 = cost_oracle.quadratic_cost(K_prior, A, B, V, episode_length=1)

        factor = 1 / settings['mixing_time']
        J_pi = J_pi_K0 * factor + (1 - factor) * J_K0

        # excitation
        data_for_exc['excitation'] = True
        data_for_exc['N'] = N
        data_for_exc['steps'] = N
        data_for_exc['cost'] = J_pi * N
        data_for_exc['cost_mean'] = J_pi

        data_for_exc['cost_sup'] = (J_pi / J_K_opt_true) * (N / T)

        # Exploitation
        data_for_exp = {}

        J_K1 = cost_oracle.quadratic_cost(K, A, B, V, episode_length=1)

        # excitation
        data_for_exp['excitation'] = False
        data_for_exp['N'] = N
        data_for_exp['steps'] = T - N
        data_for_exp['cost'] = J_K1 * (T - N)
        data_for_exp['cost_mean'] = J_K1
        data_for_exp['cost_sup'] = (J_K1 / J_K_opt_true) * ((T - N) / T)

        data.append(data_for_exc)
        data.append(data_for_exp)

    data = pd.DataFrame.from_dict(data)

    # set the figure size
    # plt.figure(figsize=(10, 7))

    # print(data)
    total = data.groupby(['N'])['cost_sup'].sum().reset_index()
    # print(total)

    # bar chart 1 -> top bars (group of 'smoker=No')
    bar1 = sns.barplot(ax=axis, x="N", y="cost_sup", data=total, saturation=.5)

    # Define some hatches
    hatches = ['x' for i in range(20)]

    # Loop over the bars
    for i, thisbar in enumerate(bar1.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])

    # bottom bar ->  take only smoker=Yes values from the data
    # bottom bar ->  take only smoker=Yes values from the data
    excitation = data[data.excitation == False]

    # bar chart 2 -> bottom bars (group of 'smoker=Yes')
    bar2 = sns.barplot(ax=axis, x="N", y="cost_sup", data=excitation, estimator=sum, ci=None)

    # add legend
    # top_bar = mpatches.Patch(color='darkblue', label='excitation = Yes')
    # bottom_bar = mpatches.Patch(color='lightblue', label='excitation = No')
    # plt.legend(handles=[top_bar, bottom_bar])

    handles, labels = bar1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # bar2.set_ylabel(
    #     r'$\frac{\mathbb{E}[ J_{\pi_e}(K_i) ] + \mathbb{E}[ J(K_{i+1}, \theta) ]}{\mathbb{E}[ J_{opt})]}$')
    # # sns.move_legend(

    bar2.set_ylabel(r'normalized $J$')
    # bar2.set_?
    #     bar1, "lower center",
    #     bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
    # )
    # show the graph
    # plt.show()

    return data

def plot_1d_result(result):

    import matplotlib as mpl
    from utils.postprocessing_utils import initialize_plot, set_size

    # Use the pgf backend (must be set before pyplot imported)
    # mpl.use('pgf')

    # You can also load latex packages
    mpl.rcParams.update({
        "pgf.preamble": '\\usepackage[utf8x]{inputenc}\\usepackage[light]{kpfonts}\\usepackage{amsfonts}\\usepackage{amsmath}\\usepackage{amssymb}',
    })

    import seaborn as sns
    import matplotlib.pyplot as plt

    settings = result['settings']

    system = LinearSystem(settings['system']['A'], settings['system']['B'], settings['system']['V'])
    K_prior = result['K0']
    controller = StateFeedbackLaw(K=K_prior)
    system.controller = controller

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

    prior_plot = sns.kdeplot(data=systems_prior, x="A", y="B", fill=True)
    plt.plot(system.A, system.B, 'rx', ms=10)
    plt.show()

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

        plot = sns.kdeplot(ax=axis, data=prior, x="A", y="B", levels=10, fill=True, color=color_prior, alpha=.5)
        plot = sns.kdeplot(ax=plot, data=post, x="A", y="B", levels=10, fill=True, color=color_post)
        # plot.set_xlim(0.88, 1.12)
        # plot.set_ylim(0.18, 1.82)
        plot.plot(system.A, system.B, 'x', mew=2.5, ms=8, color=color_true)
        # plt.show()

        return plot

    from matplotlib.gridspec import GridSpec

    # ADJUST PATH IN "initialize_plot"
    c, params = initialize_plot('CDC_paper')  # specify font size etc.,
    plt.rcParams.update(params)

    # CDC column width is 245pt, for double column plot change to 505pt
    x, y = set_size(505,
                    subplots=(1, 1),  # specify subplot layout for nice scaling
                    fraction=1.)  # scale width/height

    colors = sns.color_palette("deep", 5)
    sns.set_palette(colors)
    sns.set_style("whitegrid")
    sns.set_context("paper", rc={"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})

    # def format_axes(fig):
    #     for i, ax in enumerate(fig.axes):
    #         ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
    #         ax.tick_params(labelbottom=False, labelleft=False)


    fig = plt.figure(figsize=(x, y*0.8))


    gs = GridSpec(7, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0:3, 0:2])
    ax2 = fig.add_subplot(gs[0:3, 2:4])
    ax3 = fig.add_subplot(gs[0:3, 4:6])

    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax4 = fig.add_subplot(gs[3:7, 0:3])
    ax5 = fig.add_subplot(gs[3:7, 3:6], sharey=ax4)

    prior_samples = data_list

    # --- small
    plt_t4 = plot_1d_ussm(ax1, result['post_ussm'][1]['ussm'], prior_samples, color_prior=colors[0], color_post=colors[1], color_true=colors[4])
    # --- N_Opt
    plot_1d_ussm(ax2, result['post_ussm'][2]['ussm'], prior_samples, color_prior=colors[0], color_post=colors[2], color_true=colors[4])

    # -- Big
    plt_t2 = plot_1d_ussm(ax3, ussm=result['post_ussm'][3]['ussm'], prior_list=prior_samples, color_prior=colors[0], color_post=colors[3], color_true=colors[4])

    ax2.set(ylabel=None)
    ax2.set_yticklabels([])
    ax3.set(ylabel=None)
    ax3.set_yticklabels([])

    plot_cost_bars(ax4, result)


    n_plot = beta_data['n_plot']
    excitation = beta_data['excitation'] / (J_K_opt_true * T)
    fitted_cost = beta_data['fitted_cost'] / (J_K_opt_true * T)
    beta_parameter = beta_data['beta_parameter']

    from betl.excitation_strategy import parameterized_beta
    beta = lambda n : parameterized_beta(n, *beta_parameter)

    palette = sns.cubehelix_palette(light=.8, n_colors=3)
    ax5.plot(n_plot, excitation, color=palette[0], label=r'$N \mathbb{E}[J_{\pi_e}(K_0)]$')
    ax5.plot(n_plot, fitted_cost, color=palette[1], label=r'$(N-\Delta) \mathbb{E}[J(K_N)]$')

    # plt.plot(x, JNs.T, '--', linewidth=1)
    # plt.plot(x, (JNs + np.array(excitation_samples)).T, '--', linewidth=1)

    ax5.plot(n_plot, np.array(excitation) + np.array(fitted_cost), color=palette[2],
              label=r'$\mathbb{E}[J_{\Delta}(N)]$')


    n_list = np.array(list(N_s))
    examples = np.array(list(beta(n) for n in n_list))

    cost_reduction = examples * E_G
    total = n_list * E_J_pi + (T - n_list) * (E_J_K0 - cost_reduction)
    total = total / (J_K_opt_true * T)

    ax4.text(0.5, 0.925, r'$\mathbb{E}[J_{\Delta}(N) \mid \theta] \;\;$', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

    ax5.scatter(n_list, total, color=colors[0:4])
    ax5.set_xlabel(r'number of excitation steps $N$')
    ax5.set_ylabel('normalized expected cost')

    ax1.set_xlabel("$A$")
    ax1.set_ylabel("$B$")

    ax2.set_xlabel("$A$")
    ax3.set_xlabel("$A$")

    ax4.set_xlabel("$N$")

    ax5.set(ylabel=None)
    ax5.tick_params('y', labelleft=False)

    ax5.legend(loc='lower right')

    gs.update(left=0.065, right=0.99, top=0.95, bottom=0.12, wspace=0.1, hspace=4.5)

    # plt.savefig('1d-example-300dpi.eps', dpi=300)
    # plt.savefig('1d-example-300dpi.pdf', dpi=300)

    plt.show()
    # plt.savefig('figures/1d-example.pgf', format='pgf')
    # plt.close()

if __name__ == "__main__":
    np.random.seed(2)
    import pickle

    with open(f'data/1d-result-data.pickle', 'rb') as file:
        result = pickle.load(file)

    plot_1d_result(result)
