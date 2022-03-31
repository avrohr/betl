import numpy as np
import matplotlib.pyplot as plt
from betl.linear_system import GraphLaplacian3D

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

    settings = {
        'Q': np.eye(system.state_dimension) * 1.,
        'R': np.eye(system.input_dimension) * 5.,
        'system': {
            'A': A,
            'B': B,
            'V': V,
        },
        'prior': {
            'S0': None,
            'L0': None,
            'V0': V,
            'v0': 100,
            'independent_noise': False
        },
        'excitation_variance': 0.25 * np.eye(system.input_dimension),
        'mixing_time': 100,
        'T': 10000,  # Episode length
        'system_samples': 10,  # Number of system for approximating beta
        'plot_beta': True
    }
    synthesis_settings = {
        'confidence_interval': .99,
    }


    ussm_prior, K_prior = learn_robustly_stabilize_prior(system, settings, synthesis_settings)

    system.controller.K = K_prior
    result = cost_optimal_belief_update(ussm_prior, K_prior, system=system, settings=settings,
                                        synthesis_settings=synthesis_settings)

    print('-----------------------------------------------------------')
    from betl.plot_1d_cost_optimal_learning import plot_cost_bars

    # ax = plt.figure().gca()
    # plot_cost_bars(ax, result)
    #
    # plt.show()