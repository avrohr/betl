import numpy as np


def spectral_radius(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    return np.max(np.abs(np.linalg.eigvals(A)))


def check_stability(A, eps=1e-6):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    return spectral_radius(A) < 1 - eps


# P(empirical - E(unstable) >= eps) <= exp(-2 n eps**2) <= alpha
# n >= (1/(2eps**2)) * log(1/alpha) with alpha in (0,1), eps > 0
def one_sided_hoeffding_sample_bound(alpha=0.001, eps=0.005):
    return int(np.ceil(1./(2 * eps**2) * np.log(1/alpha)))


class LinearStabilityAnalysis:

    def __init__(self, uncertainStateSpaceModel, K, settings):
        self.K = K
        self.ussm = uncertainStateSpaceModel
        self.V = self.ussm.omega_var

        self.controllability_tol = settings.synthesis_settings['controllability_tol']

        self.confidence_interval = settings.synthesis_settings['confidence_interval']

        self.As = list()
        self.Bs = list()

    # Checks the probability of being stable w.r.t. the distribution of the state space model uncertainty
    # P(empirical - E(unstable) >= eps) <= exp(-2 n eps**2) <= alpha
    def p_stability(self, alpha=0.001, eps=0.01):

        sample_list = list()

        samples = one_sided_hoeffding_sample_bound(alpha=alpha, eps=eps)
        print('Check stability based on {} samples'.format(samples))

        if len(self.As) < samples:
            As, Bs = self.ussm.sample(n=samples,
                                      c=self.confidence_interval,
                                      controllability_tol=self.controllability_tol)
            self.As, self.Bs = As, Bs

        for A, B in zip(self.As, self.Bs):

            A_cl = A + B @ self.K
            stable = check_stability(A_cl)
            sample_list.append(stable)

        return np.mean(sample_list)
