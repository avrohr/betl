import numpy as np
import matplotlib.pyplot as plt
from betl.linear_system import GraphLaplacian3D

from betl.linear_system import StateFeedbackLaw
from betl.cost_optimal_belief_update import cost_optimal_belief_update, learn_robustly_stabilize_prior

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARN)



# import pickle
#
# with open(f'3d-result-data.pickle', 'wb') as file:
#     pickle.dump(result, file)

if __name__ == '__main__':
    np.random.seed(1)

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
    }
    synthesis_settings = {
        'confidence_interval': .99,
    }


    ussm_prior, K_prior = learn_robustly_stabilize_prior(system, settings, synthesis_settings)

    system.controller = StateFeedbackLaw(K=K_prior)
    result = cost_optimal_belief_update(ussm_prior, K_prior, system=system, settings=settings,
                                        synthesis_settings=synthesis_settings)

    print('-----------------------------------------------------------')
    from betl.plot_1d_cost_optimal_learning import plot_cost_bars

    ax = plt.figure().gca()
    plot_cost_bars(ax, result)

    plt.show()