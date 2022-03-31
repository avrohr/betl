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

np.random.seed(42)

A = np.array([[1.01]])
B = np.array([[1.]])
V = np.eye(1) * 0.02

system = LinearSystem(A, B, V)

settings = {
    'Q': np.eye(system.state_dimension) * 1e-3 * 100,
    'R': np.eye(system.input_dimension) * 1. * 100,
    'system': {
        'A': A,
        'B': B,
        'V': V,
    },
    'prior': {
        'S0': np.hstack((A, B)).T,
        'L0': np.linalg.inv(0.5 * (1.4 * np.eye(2) - .4 * np.ones((2, 2)))),
        'V0': V,
        'v0': 100,
        'independent_noise': False
    },
    'excitation_variance': 0.02 * np.eye(system.input_dimension),
    'mixing_time': 25,
    'T': 5000,  # Episode length
    'system_samples': 25,  # Number of system for approximating beta
    'plot_beta': True
}
synthesis_settings = {
    'confidence_interval': .99,
}
from betl.cost_optimal_belief_update import cost_optimal_belief_update, learn_robustly_stabilize_prior

ussm_prior, K_prior = learn_robustly_stabilize_prior(system, settings, synthesis_settings, data_per_traj=100)

A, B, V = ussm_prior.sample(n=1, c=.4)

# Sample from paper
A = [np.array([[1.04023356]])]
B = [np.array([[0.91231085]])]
V = [np.array([[0.01865229]])]
settings['system'] = {'A': A[0], 'B': B[0], 'V': V[0]}

system = LinearSystem(A[0], B[0], V[0])
system.controller = StateFeedbackLaw(K_prior)

result = cost_optimal_belief_update(ussm_prior, K_prior, system=system, settings=settings, synthesis_settings=synthesis_settings)

import pickle

with open(f'data/1d-result-data.pickle', 'wb') as file:
    pickle.dump(result, file)


print('-----------------------------------------------------------')

from betl.plot_1d_cost_optimal_learning import plot_1d_result

plot_1d_result(result)