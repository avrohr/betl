import copy

import numpy as np
from numpy.linalg import inv
from scipy.stats import matrix_normal, invwishart, chi2
from scipy.linalg import kron

import logging


def controllable(A, B, tol=None):
    n = np.shape(A)[0]
    C = B
    for i in range(1, n):
        C = np.hstack((C, (A ** i) @ B))

    return np.linalg.matrix_rank(C, tol=tol) == n


class UncertainStateSpaceModel:

    eps = 0.

    def __init__(self, prior, dim):

        # This is basically for documentation purposes...
        assert isinstance(prior, MatrixNormal)
        self.dim = dim
        n_states = self.dim[0]
        n_inputs = self.dim[1]
        assert prior.dim == (n_states+n_inputs, n_states)

        self.dist = prior
        self.prior_dist = copy.deepcopy(prior)

    @property
    def prior(self):
        return UncertainStateSpaceModel(self.prior_dist, dim=self.dim)

    def sample(self, n, c=None, controllability_tol=1e-5):

        n_states = self.dim[0]
        n_inputs = self.dim[1]

        A_c = list()
        B_c = list()
        V_c = list()

        i = 0
        max_iter = 10
        while len(A_c) < n and i < max_iter:
            i += 1

            if c is None:
                samples = self.dist.sample(n=(n-len(A_c)))
            else:
                samples = self.dist.sample_truncated(n=(n-len(A_c)), c=c)
            n_states = self.dim[0]
            n_inputs = self.dim[1]

            As = map(lambda S: S['matrix'].T[:, 0:n_states], samples)
            Bs = map(lambda S: S['matrix'].T[:, n_states:n_states + n_inputs], samples)
            Vs = map(lambda V: V['noise'], samples)


            As = list(As)
            Bs = list(Bs)
            Vs = list(Vs)

            for j in range(len(As)):
                A = As[j]
                B = Bs[j]
                V = Vs[j]

                if controllability_tol is None or controllable(A, B, tol=controllability_tol):
                    A_c.append(A)
                    B_c.append(B)
                    V_c.append(V)


        logging.info('Sampled {0} systems in {1} iterations. Wanted {2}'.format(len(A_c), i, n))

        return A_c, B_c, V_c

    def mean(self):
        S_m = self.dist.mean().T

        n_states = self.dim[0]
        n_inputs = self.dim[1]

        return S_m[:, 0:n_states], S_m[:, n_states:n_states + n_inputs]

    def var(self):
        S_var = self.dist.var().T

        n_states = self.dim[0]
        n_inputs = self.dim[1]

        return S_var[:, 0:n_states], S_var[:, n_states:n_states + n_inputs]

    def std(self):

        return np.sqrt(self.var())

    def noise_var(self):
        return np.atleast_2d(self.dist.V)


class MatrixNormal:

    eps = 0.

    def __init__(self, dim, V0=0.01, v0=None, L0=None, S0=None, independent_noise=False):

        self.dim = dim
        n_predictor = self.dim[0]
        n_dependent = self.dim[1]

        self.independent_noise = independent_noise

        # Prior for the noise covariance is an inverse Wishart distribution
        if v0 is None:
            variance_prior_scale = 1
            v0 = n_dependent + 2 + variance_prior_scale
        assert v0 > n_dependent + 1
        self.v0 = v0
        self.vn = v0

        if isinstance(V0, float):
            V0 = np.eye(n_dependent) * V0
        V0 = V0 * (self.v0 - n_dependent - 1)
        assert V0.shape == (n_dependent, n_dependent)

        self.V0 = V0
        self.Vn = V0

        # Prior for the matrix normal
        if S0 is None:
            S0 = np.eye(n_dependent, n_predictor).T
        assert S0.shape == (n_predictor, n_dependent)
        self.S0 = S0
        self.Sn = S0

        if L0 is None:
            L0 = np.eye(n_predictor, n_predictor)
        assert L0.shape == (n_predictor, n_predictor)

        self.L0 = L0
        self.Ln = L0

        # Expected parameters of the multivariate normal
        self.E = None  # Covariance
        self.E_inv = None
        self.V = None #

        self.update_multivariate_normal()

    # TODO Convenience for bound prediction... not correct
    def update_multivariate_normal(self):
        U = inv(self.Ln)
        self.V = np.atleast_2d(invwishart(df=self.vn, scale=self.Vn).mean())
        if self.independent_noise:
            self.V = np.diag(np.diag(self.V))
        self.E = np.kron(U, self.V)
        self.E_inv = inv(self.E)


    def sample(self, n=1):

        En_samples = invwishart(df=self.vn, scale=self.Vn).rvs(size=n).reshape(-1, self.dim[1], self.dim[1])

        samples = list()
        for En in En_samples:

            if self.independent_noise:
                En = np.diag(np.diag(En))

            matrix = matrix_normal(mean=self.Sn, rowcov=inv(self.Ln), colcov=En).rvs()
            samples.append({'matrix': matrix, 'noise': En})

        return samples

    def mean(self):
        return self.Sn

    # Expected variance (depends on the distribution of V
    def var(self):
        return np.diag(self.cov()).reshape(self.dim)

    # Expected covariance (depends on the distribution of V
    def cov(self):
        U = inv(self.Ln)
        V = np.atleast_2d(self.V)

        return kron(U, V)

    def update_posterior(self, X, Y):

        assert(X.shape[0] == Y.shape[0])
        assert(X.shape[1] == self.dim[0])
        assert(Y.shape[1] == self.dim[1])

        n = X.shape[0]

        Sn = inv(X.T @ X + self.Ln) @ (X.T @ Y + self.Ln @ self.Sn)
        Ln = X.T @ X + self.Ln

        vn = self.vn + n
        Vn = self.Vn + (Y - X @ Sn).T @ (Y-X @ Sn) + (Sn - self.Sn).T @ self.Ln @ (Sn - self.Sn)

        self.Sn = Sn
        self.Ln = Ln
        self.vn = vn
        self.Vn = Vn
        self.update_multivariate_normal()

        # forgetting_factor = 0.0
        # factor = (1-forgetting_factor) ** n
        # self.Sn = self.S0 * (1-factor) + Sn * factor
        # self.Ln = self.L0 * (1-factor) + Ln * factor
        # self.vn = self.v0 * (1-factor) + vn * factor
        # self.Vn = self.V0 * (1-factor) + Vn * factor

    # This is the confidence bound for a standard multivariate normal distribution
    def get_ellipse_bound(self, c=0.95):

        m = self.Sn.flatten().reshape(-1, 1)
        n = m.shape[0]

        rv = chi2(n)
        bound = rv.ppf(c)

        return bound

    # Using the expected confidence bound w.r.t. the distribution over V
    # TODO: Calculate the marginal confidence bound
    def is_inside_conf(self, S, bound):

        x = S.flatten().reshape(-1, 1)
        m = self.Sn.flatten().reshape(-1, 1)
        return (x - m).T @ self.E_inv @ (x - m) <= bound

    def sample_truncated(self, n=1, c=0.95):

        accepted_samples = list()
        bound = self.get_ellipse_bound(c)

        while len(accepted_samples) < n:

            # print(len(accepted_samples))
            expected_draws = int(((n - len(accepted_samples)) * (1/c)))
            samples = self.sample(min(max(10, expected_draws), int(1e6)))

            for sample in samples:
                if len(accepted_samples) < n and self.is_inside_conf(sample['matrix'], bound=bound):
                    accepted_samples.append(sample)

        return np.array(accepted_samples)


if __name__ == '__main__':
    test = MatrixNormal((3,2), V0=0.02)

    A = np.array([[0.9, 0.2, 0.],
                  [0.0, 0.8, 0.2]])

    n = 2
    m = 2

    def get_trajectory(x0, n=1000):
        x = np.zeros((2, n))
        x[:,[0]] = x0
        u = np.random.rand(1,n)
        for i in range(n-1):
            x[:, [i+1]] = A @ np.vstack((x[:,[i]], u[:,[i]])) + np.random.randn(2, 1) * np.sqrt(0.01)

        return x, u

    T_prior = get_trajectory(np.array([[0, 0]]).T, n=10)

    for i in range(10):
        x, u = get_trajectory(np.array([[0, 0]]).T, n=500)

        X = np.vstack((x[:, 10:-1], u[:, 10:-1])).T
        Y = x[:, 11:].T

        test.update_posterior(X, Y)

        #print(test.mean())


    print(test.var())
    print(test.sample())

    ussm = UncertainStateSpaceModel(prior=test, dim=(2, 1))
    test = ussm.sample(n=2, c=0.50, controllability_tol=None)
    print(test)