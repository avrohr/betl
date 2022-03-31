import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from betl.linear_system import DiscreteTimeLinearSystem as LinearSystem
from betl.linear_system import GraphLaplacian3D
from betl.linear_system import StateFeedbackLaw, ExcitingStateFeedbackLaw
from betl.synthesis.robust_lqr_synth import RLQRSyntheziser

from betl.uncertain_state_space_model import MatrixNormal, UncertainStateSpaceModel
from betl.cost_analysis import LinearQuadraticCostAnalysis, EmpiricalQuadraticCostAnalysis

from betl.excitation_strategy import optimal_signal
from scipy.linalg import solve_discrete_are

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARN)

def learn_robustly_stabilize_prior(system, settings, synthesis_settings, data_per_traj=5):

    A, B, V = system.A, system.B, system.V

    Q = settings['Q']
    R = settings['R']
    P = np.array(np.array(solve_discrete_are(A, B, Q, R)))
    K_ex = - np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)


    dist = MatrixNormal((system.state_dimension + system.input_dimension, system.state_dimension),
                        L0=settings['prior']['L0'],
                        S0=settings['prior']['S0'],
                        V0=settings['prior']['V0'],
                        v0=settings['prior']['v0'],
                        independent_noise=False)

    ussm = UncertainStateSpaceModel(dist, (system.state_dimension, system.input_dimension))

    # ---  Create feasible prior
    i = 0
    synth = RLQRSyntheziser(uncertainStateSpaceModel=ussm, settings=synthesis_settings, Q=Q, R=R)
    K = synth.synthesize()
    while K is None:
        i = i + 1

        system.controller = StateFeedbackLaw(K=K_ex)
        x0s = system.sample_steady_state(n=data_per_traj)

        system.controller = ExcitingStateFeedbackLaw(covariance=settings['excitation_variance'] * 10., K=np.zeros_like(K_ex))
        data = system.sample_trajectories(x0s=x0s, length=1)

        x = data['predictor']
        y = data['dependent']
        ussm.dist.update_posterior(X=x.T, Y=y.T)

        synth = RLQRSyntheziser(uncertainStateSpaceModel=ussm, settings=synthesis_settings, Q=Q, R=R)
        K = synth.synthesize()

    print('Trajectories for prior {}'.format(i))
    K_prior = K

    # Determine optimal excitation
    ussm_prior = copy.deepcopy(ussm)
    ussm_prior.prior_dist = copy.deepcopy(ussm.dist)
    return ussm_prior, K_prior

def cost_optimal_belief_update(ussm_prior, K_prior, system, settings, synthesis_settings):
    T = settings['T']
    Q = settings['Q']
    R = settings['R']

    A, B, V = system.A, system.B, system.V

    P = np.array(np.array(solve_discrete_are(A, B, Q, R)))
    K_opt = - np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    cost_oracle = LinearQuadraticCostAnalysis(ussm_prior, Q, R)

    # Controller cost on the true (unknown) system

    J_K0_true = cost_oracle.quadratic_cost(K_prior, A, B, V, episode_length=1)
    J_pi_true = cost_oracle.quadratic_cost_with_input(K_prior, A, B, V,
                                                      input_variance=settings['excitation_variance'],
                                                      episode_length=1)
    J_K_opt_true = cost_oracle.quadratic_cost(K=K_opt, A=A, B=B, V=V, episode_length=1)

    # Controller cost on the known belief over system parameter

    J_K0 = cost_oracle.expected_cost(K_prior, episode_length=1, samples=132, c=synthesis_settings['confidence_interval'])
    G = cost_oracle.expected_optimality_gap(K_prior, episode_length=1, samples=132, c=synthesis_settings['confidence_interval'])
    J_pi = cost_oracle.expected_cost(K_prior, episode_length=1, samples=132,
                                     c=synthesis_settings['confidence_interval'],
                                     input_variance=settings['excitation_variance'])

    E_J_K0 = np.mean(J_K0)
    E_G = np.mean(G)
    factor = 1/settings['mixing_time']
    E_J_pi = np.mean(J_pi) * factor + (1 - factor) * E_J_K0

    print('Expected cost for the current controller {}'.format(E_J_K0))
    print('Expected cost including the excitation signal {}+-{}'.format(E_J_pi, np.std(J_pi)))
    print('Additional cost from learning {} and expected gap {}'.format(E_J_pi - E_J_K0, E_G))

    N_opt, beta_data = optimal_signal(ussm_prior, K_prior, E_G, E_J_K0, E_J_pi, synthesis_settings, settings)

    print('N_opt {}'.format(N_opt))


    def update_belief(N, ussm_prior, K_prior, system, x0, settings):

        A, B, V = system.A, system.B, system.V

        controller = ExcitingStateFeedbackLaw(K=K_prior, covariance=settings['excitation_variance'])
        system.controller = controller

        data = system.sample_trajectories(x0s=x0, length=N*settings['mixing_time'])
        x = data['predictor'][::, ::settings['mixing_time']]
        y = data['dependent'][::, ::settings['mixing_time']]

        assert y.shape[1] == N

        ussm_post = copy.deepcopy(ussm_prior)
        ussm_post.dist.update_posterior(X=x.T, Y=y.T)

        synth = RLQRSyntheziser(uncertainStateSpaceModel=ussm_post, settings=synthesis_settings, Q=Q, R=R)
        K_post = synth.synthesize()

        cost_oracle = LinearQuadraticCostAnalysis(ussm_post, Q, R)

        J_K1 = cost_oracle.expected_cost(K=K_post, episode_length=1, samples=132, c=synthesis_settings['confidence_interval'])
        E_J_K1 = np.mean(J_K1)
        print('Using {} samples'.format(N))

        print('Expected cost before {} and after {}'.format(E_J_K0, E_J_K1))

        true_J_K0 = cost_oracle.quadratic_cost(K=K_prior, A=A, B=B, V=V)
        true_J_K1 = cost_oracle.quadratic_cost(K=K_post, A=A, B=B, V=V)
        print('true cost before {} and after {}'.format(true_J_K0, true_J_K1))

        result = {
            'N': N,
            'ussm': ussm_post,
            'E_J_K1': E_J_K1,
            'K1': K_post,
            'true_J_K0': true_J_K0,
            'true_J_K1': true_J_K1
        }
        return result


    result = dict()
    N_s = list((0, int(T/50), N_opt, int(T/2)))

    system.controller = StateFeedbackLaw(K_prior)
    x0 = system.sample_steady_state(n=1)
    print(x0)

    result['prior_ussm'] = ussm_prior
    result['K0'] = K_prior
    result['post_ussm'] = list()

    for N in N_s:
        __result = update_belief(N, ussm_prior, K_prior, system, x0, settings)
        __result['optimal'] = N == N_opt
        result['post_ussm'].append(__result)


    K_s = list()
    K_s += list(result['K1'] for result in result['post_ussm'])

    result['Ns'] = N_s
    result['Ks'] = K_s

    result['beta_data'] = beta_data

    result['settings'] = settings
    result['synthesis_settings'] = synthesis_settings

    result['J_K0_true'] = J_K0_true
    result['J_pi_true'] = J_pi_true
    result['J_K_opt_true'] = J_K_opt_true
    result['J_K0'] = J_K0
    result['G'] = G
    result['J_pi'] = J_pi
    result['E_J_K0'] = E_J_K0
    result['E_G'] = E_G
    result['factor'] = factor
    result['E_J_pi'] = E_J_pi

    return result