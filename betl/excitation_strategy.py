import numpy as np
import matplotlib.pyplot as plt
import copy
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.optimize import curve_fit

import seaborn as sns

import pandas as pd

from betl.linear_system import DiscreteTimeLinearSystem
from betl.linear_system import StateFeedbackLaw, ExcitingStateFeedbackLaw
from betl.cost_analysis import LinearQuadraticCostAnalysis
from betl.synthesis.robust_lqr_synth import RLQRSyntheziser

# def parameterized_beta(x, n_beta, GN):
#          f = interp1d(n_beta, GN)
#          return f(x)


def parameterized_beta(x, a, b, c, d, e):
    exp = e * (1 - np.power(b, -a * x)) + (1-e) * (1 - np.power(c, -d * x))
    # sigm = 1/(1 + np.exp(-g*x - f))
    return exp

def optimal_signal(ussm, K, E_G, E_J_K0, E_J_pi, synthesis_settings, settings):

    data = dict()

    params, _, _, result_beta_est = estimate_beta(ussm, K, synthesis_settings, settings)

    # Plot total cost
    fitted_cost = list()
    excitation = list()

    T = settings['T']

    n_plot = range(0, T)
    for n in n_plot:

        beta = parameterized_beta(n, *params)

        cost = E_J_K0 - beta * E_G
        fitted_cost.append(cost * (T-n))
        excitation.append(n * E_J_pi)

    data['n_plot'] = n_plot
    data['excitation'] = excitation
    data['fitted_cost'] = fitted_cost
    data['beta_parameter'] = params
    data['result_beta_est'] = result_beta_est

    beta_f = lambda n : parameterized_beta(n, *params)
    # Find N_opt
    def F(x):

        x = x
        cost_reduction = beta_f(x) * E_G
        total = x * E_J_pi + (T - x) * (E_J_K0 - cost_reduction)

        return total

    bnds = ((0, T),)
    sol = minimize(F, x0=np.array([T/2.]), bounds=bnds)

    N_opt = int(np.ceil(sol.x)[0])

    return N_opt, data

#Estimates the 'improvement rate' of the current Bayesian model
def estimate_beta(ussm, K, synthesis_settings, settings):

    T = settings['T']

    n_beta = np.array([int(i) for i in np.logspace(np.log10(T/10), np.log10(T), 20, endpoint=True)])
    system_samples = settings['system_samples']
    n_beta = np.hstack((np.array([0]), n_beta))

    As, Bs, Vs = ussm.sample(n=system_samples, c=synthesis_settings['confidence_interval'])
    #As, Bs, Vs = ussm.sample(n=system_samples, c=0.1)

    # system_samples = 1
    # A, B = ussm.mean()
    # V = ussm.var()
    # #
    # As = list(A)
    # Bs = list(B)
    # Vs = list(V)

    result = Parallel(n_jobs=5)(delayed(estimate_J_N)
                                (A=np.atleast_2d(As[j]), B=np.atleast_2d(Bs[j]), V=np.atleast_2d(Vs[j]),
                                 ussm_prior=ussm, Ns=n_beta, K_prior=K,
                                 synthesis_settings=synthesis_settings, settings=settings)
                                for j in range(system_samples))
    # flatten
    result = list(item for sublist in result for item in sublist)
    result = pd.DataFrame.from_dict(result)

    mean_GN = result.groupby('N')['G_N'].mean()
    mean_JN = result.groupby('N')['J_N'].mean()
    GN = np.array(mean_GN.values)

    GNs = result.groupby('N')['G_N'].apply(np.hstack).values
    GNs = np.array(list(ar for ar in GNs))

    y_data = GNs.flatten()
    x_data = n_beta.repeat(system_samples)

    JNs = np.array(mean_JN.values)
    bnds = ([0., 0.0, 0.0, 0.0, 0.0],
             [np.inf, np.inf, np.inf, np.inf, 1.])
    beta_parameter, pcov_exp = curve_fit(parameterized_beta, x_data, y_data, bounds=bnds)
    beta = lambda N: parameterized_beta(N, *beta_parameter)
    print("beta_parameter")
    print(beta_parameter)
    #beta = interp1d(n_beta, GN)

    #beta = lambda n: kr.predict(np.atleast_2d(n))

    if settings['plot_beta']:
        # Plot beta

        GNs = result.groupby('N')['G_N'].apply(np.hstack).values
        GNs = np.array(list(ar for ar in GNs))

        beta_plot = np.linspace(0, T, T)

        G_param = list(beta(n) for n in beta_plot)
        G_param = np.array(G_param).flatten()

        sns.set_style("whitegrid")
        sns.set_context("paper")

        mm = 1 / 25.4  # centimeters in inches
        fig = plt.figure(figsize=(120 * mm, 142.5/2 * mm), constrained_layout=True)

        col = sns.color_palette("deep")
        grey = col[7]
        blue = col[0]
        plt.scatter(x_data, y_data, label='MC samples', marker='+', s=2, color=grey)
        plt.plot(beta_plot, G_param, label='fitted', linewidth=2, color=blue)
        plt.plot(n_beta[1:], GN[1:], label='mean', linewidth=1, color=grey)
        # splt.fill_between(n_beta, GN-np.std(GNs, axis=1), GN+np.std(GNs, axis=1), facecolor='#386B94', alpha=0.5)

        axes = plt.gca()
        axes.legend(loc='center right')

        axes.set_xlabel(r'number of excitation steps $N$')
        axes.set_ylabel(r'$\beta$')

        data_for_plots = dict()

        data_for_plots['x_data'] = x_data
        data_for_plots['y_data'] = y_data
        data_for_plots['beta_plot'] = beta_plot
        data_for_plots['n_beta'] = n_beta
        data_for_plots['G_param'] = G_param
        data_for_plots['GNs'] = GNs
        data_for_plots['GN'] = GN
        data_for_plots['beta_parameter'] = beta_parameter

        # import pickle
        #
        # with open('data/beta_data_for_plots.pickle', 'wb') as handle:
        #     pickle.dump(data_for_plots, handle)
        #
        # result.to_pickle('data/beta_dataframe.pickle')


        sns.despine(fig=fig)
        # plt.savefig('beta-example-300dpi.eps', dpi=300)

        plt.show()

    return beta_parameter, GNs, JNs, result

# Estimates the cost of a controller with an posterior based on a fixed excitation signal length
# Takes care of copying the ussm
def estimate_J_N(A, B, V, ussm_prior, K_prior, Ns, synthesis_settings, settings):

    Q = settings['Q']
    R = settings['R']
    excitation_variance = settings['excitation_variance']

    print('Estimate JNs')
    J_K_0 = LinearQuadraticCostAnalysis(ussm_prior, Q, R).quadratic_cost(K=K_prior, A=A, B=B, V=V, episode_length=1)
    G_0 = LinearQuadraticCostAnalysis(ussm_prior, Q, R).optimality_gap(K=K_prior, A=A, B=B, V=V, episode_length=1)

    synth = RLQRSyntheziser(uncertainStateSpaceModel=ussm_prior, settings=synthesis_settings, Q=Q, R=R)

    estimates = list()
    N_add = np.concatenate((np.atleast_1d(Ns[0]), np.diff(Ns)))
    runs = 3
    for i in range(runs):

        J_N = J_K_0

        ussm = copy.deepcopy(ussm_prior)
        ussm.prior_dist = copy.deepcopy(ussm_prior.dist)

        N_total = 0
        for N in N_add:
            N_total += N

            expected_costs = list()
            #print('N {} - seed {}'.format(N, seed))

            if N == 0:
                pass

            else:

                system = DiscreteTimeLinearSystem(A=A, B=B, V=V)
                system.controller = ExcitingStateFeedbackLaw(K=K_prior, covariance=excitation_variance)

                x0s = system.sample_steady_state(N)
                data = system.sample_trajectories(x0s=x0s, length=1)

                x = data['predictor']
                y = data['dependent']

                # print('Creating data for N = {0}, result = {1}'.format(N, y.shape[1]))
                # print(N_total)
                ussm.dist.update_posterior(X=x.T, Y=y.T)
                synth.ussm = ussm
                K_post = synth.synthesize()

                if K_post is None:
                    print('Failed synthesis in MC with {} samples'.format(N))
                    K_post = K_prior

                J_N = LinearQuadraticCostAnalysis(ussm, Q, R).quadratic_cost(K=K_post, A=A, B=B, V=V, episode_length=1)

            estimates.append(
                {
                    'J_K_0': J_K_0,
                    'G_0': G_0,
                    'N': N_total,
                    'J_N': J_N,
                    'G_N': (J_K_0 - J_N) / G_0
                }
            )

    result = pd.DataFrame.from_dict(estimates)
    mean_result = result.groupby('N').mean().reset_index().to_dict('records')

    return mean_result