import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import picos as pic
from joblib import Parallel, delayed

from betl.linear_system import DiscreteTimeLinearSystem as LinearSystem
from betl.linear_system import GraphLaplacian3D, OneDimensionalToy
from betl.linear_system import StateFeedbackLaw, ExcitingStateFeedbackLaw
from betl.synthesis.robust_lqr_synth import RLQRSyntheziser

from betl.uncertain_state_space_model import MatrixNormal, UncertainStateSpaceModel
from betl.cost_analysis import LinearQuadraticCostAnalysis, EmpiricalQuadraticCostAnalysis

from betl.excitation_strategy import optimal_signal
from betl.cost_optimal_belief_update import learn_robustly_stabilize_prior
from chernoff_bounds import bounds as chernoff_bounds

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARN)


def cost_from_data(states, inputs, Q, R):
    x = states
    u = inputs

    cost = list()
    offset = 0
    for i in range(offset, u.shape[1]):

        x_i = x[:, [i]]
        u_i = u[:, [i]]

        c_i = x_i.T @ Q @ x_i + u_i.T @ R @ u_i
        cost.append(c_i)

    cost = np.array(cost)
    return cost

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')

def window_average(x, w):
    x = x.flatten()
    modulo = x.shape[0] % w
    if modulo > 0:
        x = x[:-modulo]
    assert x.shape[0] % w == 0

    x = x.reshape(-1, w)

    return np.sum(x, axis=1)

def cov_from_model(A, B, K, V):

    A_cl = A + B @ K
    d = A.shape

    X_V = pic.SymmetricVariable('X_V', shape=d)

    F = pic.Problem()

    F.set_objective('min', pic.trace(X_V))

    F.add_constraint(A_cl.T * X_V * A_cl - X_V + V == 0)
    F.add_constraint(X_V >> 0)

    F.solve(verbosity=0, primals=None)

    # Unstable, so expected variance is infinite
    if F.status != 'optimal':
        return np.Inf

    X_V = np.atleast_2d(X_V.value)

    return X_V

def sample_bounds(ussm, K, Q, R, T, system_samples, settings, synthesis_settings):

    As, Bs, Vs = ussm.sample(n=system_samples, c=synthesis_settings['confidence_interval'])

    bounds_list = list()
    state_cov_list = list()

    for j in range(system_samples):

        #print(j)

        A = As[j]
        B = Bs[j]
        V = Vs[j]

        state_cov = cov_from_model(A, B, K, V)
        state_cov_list.append(state_cov)

    bounds_list = Parallel(n_jobs=3)(delayed(chernoff_bounds)
                                     (E=state_cov_list[j], Q=Q, R=R, A=As[j], B=Bs[j],
                                      K=K, n=T, p=settings['bounds']['confidence'])
                                     for j in range(system_samples))

    # for j in range(system_samples):
    #
    #     #print(j)
    #
    #     A = As[j]
    #     B = Bs[j]
    #     V = Vs[j]
    #
    #     state_cov = cov_from_model(A, B, K, V)
    #
    #     b = chernoff_bounds(E=state_cov, Q=Q, R=R, A=A, B=B, K=K, n=T, p=settings['bounds']['confidence'])
    #
    #     bounds_list.append(b)

    return bounds_list

def simulate_and_switch(system, n, switching_times, ussm_prior):

    changed = False
    if switching_times.size > 1 and len(system.trajectory) >= switching_times[0]:
        print("Start changing system")
        delta = 1
        while abs(delta - 1) < 0.001:
            As, Bs, Vs = ussm_prior.sample(n=1, c=synthesis_settings['confidence_interval'])

            I = np.eye(system.A.shape[0])

            A_old = system.A + system.B @ system.controller.K
            _old = np.vstack((np.hstack((A_old, system.V)), np.hstack((I, np.zeros_like(system.V)))))

            A_new = As[0] + Bs[0] @ system.controller.K
            _new = np.vstack((np.hstack((A_new, Vs[0])), np.hstack((I, np.zeros_like(system.V)))))

            delta = np.linalg.norm(_new, ord=2) / np.linalg.norm(_old, ord=2)

            system.A = As[0]
            system.B = Bs[0]
            system.V = Vs[0]
            changed = True

        state_trajectory, input_trajectory = system.simulate(steps=1 * n)
        # TODO: Burn some data after change?
        print("Changed system at T={0}, switching time is {1}".format(len(system.trajectory) / settings['mixing_time'],
                                                                      switching_times[0] / settings['mixing_time']))
        print('System change is {}'.format(delta))

    state_trajectory, input_trajectory = system.simulate(steps=1 * n)

    return state_trajectory, input_trajectory, changed


# System prior

np.random.seed(42)

system = GraphLaplacian3D()
A, B, V = system.A, system.B, system.V
system.B = B * 0.1
A, B, V = system.A, system.B, system.V



S0 = np.hstack((A, B)).T
L0 = np.linalg.inv(.1 * (np.eye(6, 6) * .7 + np.ones((6, 6)) * .3))

settings = {
    'Q': np.eye(system.state_dimension) * 1e-3,
    'R': np.eye(system.input_dimension) * 1.,
    'system': {
        'A': A,
        'B': B,
        'V': V,
    },
    'prior': {
        'S0': np.hstack((A, B)).T,
        'L0': L0,
        'V0': V,
        'v0': 10,
        'independent_noise': False
    },
    'excitation_variance': 0.02 * np.eye(system.input_dimension),
    'mixing_time': 25,  # 50
    'T': 5000,  # Episode length
    'system_samples': 25,  # Number of system for approximating beta
    'plot_beta': True,
    'bounds': {
        'samples': 50,
        'confidence': 0.001,
        'window': 200
    }
}

cost_window = settings['bounds']['window']  # settings['mixing_time'] * 1

synthesis_settings = {
    'confidence_interval': .99,
}

ussm_prior, K_prior = learn_robustly_stabilize_prior(system, settings, synthesis_settings, data_per_traj=5)

As, Bs, Vs = ussm_prior.sample(n=10, c=synthesis_settings['confidence_interval'])

# Controller cost on the known belief over system parameter
cost_oracle = LinearQuadraticCostAnalysis(ussm_prior, settings['Q'], settings['R'])
J_K0 = cost_oracle.expected_cost(K_prior, episode_length=1, samples=132, c=synthesis_settings['confidence_interval'])
G = cost_oracle.expected_optimality_gap(K_prior, episode_length=1, samples=132,
                                        c=synthesis_settings['confidence_interval'])
J_pi = cost_oracle.expected_cost(K_prior, episode_length=1, samples=132,
                                 c=synthesis_settings['confidence_interval'],
                                 input_variance=settings['excitation_variance'])

E_J_K0 = np.mean(J_K0)
E_G = np.mean(G)
factor = 1 / settings['mixing_time']
E_J_pi = np.mean(J_pi) * factor + (1 - factor) * E_J_K0
#
print('Expected cost for the current controller {}+-{}'.format(E_J_K0, np.std(J_K0)))
print('Expected cost including the excitation signal {}'.format(E_J_pi))
print('Additional cost from learning {} and expected gap {}'.format(E_J_pi - E_J_K0, E_G))

# N_opt, beta_data = optimal_signal(ussm_prior, K_prior, E_G, E_J_K0, E_J_pi, synthesis_settings, settings)
N_opt = 549
print(N_opt)
# Run 4 episodes with changing systems, detect the change

system.controller = StateFeedbackLaw(K=K_prior)
x0 = system.sample_steady_state(1)
system.reset_system(x0)

T = settings['T']
cost = list()
ps = list()

bound_list = list()
cost_list = list()

switches = np.random.randint(low=int(T), high=int(T*1.2), size=5) * settings['mixing_time']
__switches = switches
switching_times = np.cumsum(switches)
switches_actual = list()
detected = list()

model_correct = False
changed = False

cost_episode = list()
ps_local = list()
t_start_cost = len(system.trajectory)

while len(system.trajectory) < switching_times[-1]:

    if len(cost_list) > 0:
        cost_list[-1]['t_end'] = len(system.trajectory)

    t_changed = len(system.trajectory)

    if not model_correct:
        # Learn new controller
        controller = ExcitingStateFeedbackLaw(K=K_prior, covariance=settings['excitation_variance'], wait=settings['mixing_time'])
        system.controller = controller

        print('Start data collection')

        state_trajectory, input_trajectory = system.simulate(steps=N_opt * settings['mixing_time'])

        emp_cost = cost_from_data(state_trajectory, input_trajectory, Q=settings['Q'], R=settings['R'])
        emp_cost = emp_cost.flatten()
        cost_avg = window_average(emp_cost, cost_window)

        t = len(system.trajectory) - (N_opt * settings['mixing_time'])
        for cost_ in cost_avg:
            t += cost_window
            cost_episode.append(cost_)
            cost.append({
                't': t,
                'p': np.NaN,
                'cost': cost_,
                'log_cost': np.log10(cost_)
            })

        predictor = np.vstack((state_trajectory[:, 0:-1], input_trajectory[:, 0:]))
        dependent = state_trajectory[:, 1:]

        x = predictor[::, ::settings['mixing_time']]
        y = dependent[::, ::settings['mixing_time']]


        print('Finish data collection')

        ussm_post = copy.deepcopy(ussm_prior)
        ussm_post.dist.update_posterior(X=x.T, Y=y.T)

        synth = RLQRSyntheziser(uncertainStateSpaceModel=ussm_post, settings=synthesis_settings,
                                Q=settings['Q'], R=settings['R'])
        K_post = synth.synthesize()
        # Burn data
        system.simulate(steps=settings['mixing_time'])
        print('Finish new controller')

        cost_oracle = LinearQuadraticCostAnalysis(ussm_post, settings['Q'], settings['R'])
        J_KN = cost_oracle.expected_cost(K_post, episode_length=1, samples=132,
                                         c=synthesis_settings['confidence_interval'])
        E_J_KN = np.mean(J_KN)

        print('Expected cost for the updated controller {}+-{}'.format(E_J_KN, np.std(J_KN)))

        print('Start estimating bound distribution')

        system.controller = StateFeedbackLaw(K_post)
        bounds = sample_bounds(ussm_post, K_post, Q=settings['Q'], R=settings['R'],
                               T=cost_window, system_samples=settings['bounds']['samples'], settings=settings, synthesis_settings=synthesis_settings)

        print('Finish estimating bound distribution')


        lower = np.array(list(b[0] for b in bounds))
        upper = np.array(list(b[1] for b in bounds))

        bound_list.append({
            't_start_cost': t_start_cost,
            't_start': len(system.trajectory),
            'lower': lower,
            'upper': upper,
        })

        model_correct = True


    state_cov = cov_from_model(system.A, system.B, K_post, system.V)
    b = chernoff_bounds(E=state_cov, Q=settings['Q'], R=settings['R'], A=system.A, B=system.B, K=K_post, n=cost_window,
                        p=settings['bounds']['confidence'])

    prior_cost_true = cost_oracle.quadratic_cost(K=K_prior, A=system.A, B=system.B, V=system.V,
                                                 episode_length=cost_window)
    post_cost_true = cost_oracle.quadratic_cost(K=K_post, A=system.A, B=system.B, V=system.V,
                                                episode_length=cost_window)
    learning_cost_true = cost_oracle.quadratic_cost_with_input(K=K_post, A=system.A, B=system.B, V=system.V,
                                                               input_variance=settings['excitation_variance'],
                                                               episode_length=cost_window)

    cost_list.append({
        't_start': t_changed,
        'lower_true': b[0],
        'upper_true': b[1],
        'prior_cost_true': prior_cost_true,
        'post_cost_true': post_cost_true,
        'learning_cost_true': learning_cost_true,
    })

    print('True cost {0}, bounds from {1} - {2}'.format(post_cost_true, np.mean(lower), np.mean(upper)))

    changed = False
    while model_correct and not changed and len(system.trajectory) < switching_times[-1]:

        state_trajectory, input_trajectory, changed = simulate_and_switch(system=system,
                                                                          n=cost_window,
                                                                          switching_times=switching_times,
                                                                          ussm_prior=ussm_prior)
        # state_trajectory, input_trajectory = system.simulate(steps=5 * cost_window)  # TODO: parameter for horizon?
        if changed:
            print('----------------------------')

            switching_times = switching_times[1:]
            __switches = __switches[1:]
            switches_actual.append(len(system.trajectory))

        emp_cost = cost_from_data(state_trajectory, input_trajectory, Q=settings['Q'], R=settings['R'])
        emp_cost = emp_cost.flatten()
        cost_avg = window_average(emp_cost, cost_window)

        t = len(system.trajectory) - cost_window

        # print(len(system.trajectory) / settings['mixing_time'])
        for cost_ in cost_avg:

            p_l = lower > cost_
            p_u = upper < cost_

            p = np.logical_or(p_l, p_u)

            p_model_wrong = np.sum(p) / upper.shape[0]
            ps.append(p_model_wrong)
            ps_local.append(p_model_wrong)

            t += cost_window

            cost_episode.append(cost_)
            cost.append({
                't': t,
                'p': p_model_wrong,
                'cost': cost_,
                'log_cost': np.log10(cost_),
                'log_cost_above_thresh': np.log10(cost_) if p_model_wrong > 0.95 else np.NaN,
                'cost_above_thresh': cost_ if p_model_wrong > 0.95 else np.NaN

            })

        if np.sum(np.array(ps_local[-1:]) > .95) > 0.5:
            bound_list[-1]['t_end'] = len(system.trajectory)
            bound_list[-1]['cost_mean'] = np.mean(cost_episode)

            model_correct = False
            ps_local = list()
            cost_episode = list()
            t_start_cost = len(system.trajectory)

            detected.append(len(system.trajectory))
            print("Model change detected at {}".format(len(system.trajectory) / settings['mixing_time']))
            print('++++++++++++++++++++')


bound_list[-1]['t_end'] = len(system.trajectory)
cost_list[-1]['t_end'] = len(system.trajectory)
bound_list[-1]['cost_mean'] = np.mean(cost_episode)

# plt.plot(cost)
# plt.plot(switching_times / cost_window, np.ones_like(switching_times), '+')
# plt.show()

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

data = pd.DataFrame.from_dict(cost)

data_for_plots = dict()

data_for_plots['switches_actual'] = switches_actual
data_for_plots['detected'] = detected
data_for_plots['bound_list'] = bound_list
data_for_plots['cost_list'] = cost_list
data_for_plots['cost_'] = cost_


import pickle

with open('data/improved_data_for_plots.pickle', 'wb') as handle:
    pickle.dump(data_for_plots, handle)

data.to_pickle('data/improved_dataframe.pickle')


mm = 1 / 25.4  # centimeters in inches
fig = plt.figure(figsize=(140 * mm, 142.5 /2 * mm), constrained_layout=True)

cost_plot = sns.scatterplot(x="t", y="cost", data=data, marker='.', alpha=0.3, edgecolor="none", color=color_cost)
cost_plot_red = sns.scatterplot(x="t", y="cost_above_thresh", data=data, marker='o', edgecolor=color_outlier, color='none',
                            ax=cost_plot)


for time in switches_actual:
    switch = cost_plot.axvline(x=time, linewidth=1, color='k', linestyle='--')
for time in detected:
    detected_plot = cost_plot.axvline(x=time, linewidth=1, color='b', linestyle='--')

# For the legend
lc = cost_plot.hlines(y=cost_, xmin=1, xmax=1.,
                      linewidth=3, color=color_bounds, linestyle='-', alpha=1)

for bound in bound_list:

    for low in bound['lower']:

        cost_plot.hlines(y=low, xmin=bound['t_start'], xmax=bound['t_end'],
                    linewidth=3, color=color_bounds, linestyle='-', alpha=5 / 100)

    for high in bound['upper']:
        cost_plot.hlines(y=high, xmin=bound['t_start'], xmax=bound['t_end'],
                    linewidth=3, color=color_bounds, linestyle='-', alpha=5 / 100)

    emp_cost = plt.hlines(y=bound['cost_mean'], xmin=bound['t_start_cost'], xmax=bound['t_end'],
                          linewidth=2, color=color_cost_true_post, linestyle=':')

for cost_true in cost_list:
    rob_cost = plt.hlines(y=cost_true['prior_cost_true'], xmin=cost_true['t_start'], xmax=cost_true['t_end'],
                          linewidth=2, color=color_cost_true_prior, linestyle=':')
    # plt.hlines(y=cost_true['post_cost_true'], xmin=cost_true['t_start'], xmax=cost_true['t_end'],
    #            linewidth=1, color=color_cost_true_post, linestyle='--')
    # plt.hlines(y=cost_true['total_cost_true'], xmin=cost_true['t_start'], xmax=cost_true['t_end'],
    #            linewidth=1, color='black', linestyle='--')

#cost_plot.legend()
#cost_plot.legend(labels = ['change', '_no_', '_no_', '_no_', 'detection', '_no_', '_no_', '_no_', 'cost', '_no_', 'posterior bound'],
#                 handler_map={lc : HandlerLineCollection(update_func=updateline)})



cost_plot.legend(handles=[lc, switch, detected_plot, rob_cost, emp_cost], labels=['bounds', 'change', 'detection', 'robust', 'improved'])
sns.move_legend(cost_plot, "lower center", bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=True)

cost_plot.set_xlabel('time steps - $k$')
cost_plot.set_ylabel('observed cost - $J(K_i)$')

sns.despine(fig=fig)
# plt.savefig('improve-3d-300dpi.pdf', dpi=300)

plt.show()