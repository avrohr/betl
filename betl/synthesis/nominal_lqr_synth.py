import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are
import logging
from betl.synthesis.syntheziser import LQRSyntheziser


class NLQRSyntheziser(LQRSyntheziser):

    def __init__(self, uncertainStateSpaceModel, Q, R, settings):

        super().__init__(uncertainStateSpaceModel, Q, R, settings)

        self.picos_eps = 1e-9

    def synthesize(self):

        logging.info("Start computing nominal controller.")

        K = None
        try:

            A, B = self.ussm.mean()

            P = np.array(np.array(solve_discrete_are(A, B, self.Q, self.R)))
            K = - np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)

        except Exception as e:
            logging.info(e)
            logging.info('Failed computing nominal controller.')

        logging.info('Successful computing nominal controller.')

        return K
