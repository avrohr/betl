import numpy as np
import picos as pic
from scipy.linalg import solve_discrete_are

class ControlLaw:

    def __init__(self, K):
        self.K = K

    def __call__(self, x):
        raise NotImplementedError

    def to_dict(self):
        data = self.__dict__.copy()
        data['name'] = self.__class__.__name__
        return data


class StateFeedbackLaw(ControlLaw):

    def __call__(self, x):
        return self.K @ x


class ExcitingStateFeedbackLaw(ControlLaw):

    def __init__(self, covariance, K, wait=None):
        assert (covariance.shape[0] == K.shape[1])
        assert (covariance.shape[1] == K.shape[1])

        self.covariance = covariance
        super().__init__(K)

        self.counter = wait
        self.wait = wait

    def __call__(self, x):

        state_dimension = self.K.shape[1]
        input_dimension = self.K.shape[0]

        if self.wait is not None and self.counter < self.wait:
            self.counter += 1
            delta_u = 0
        else:
            delta_u = np.random.multivariate_normal(np.zeros(state_dimension), self.covariance).reshape(input_dimension, 1)
            self.counter = 1
            #return np.ones_like(delta_u)

        u = self.K @ x
        return u + delta_u


class DiscreteTimeLinearSystem(object):

    def __init__(self, A, B, V):
        self.A = A
        self.B = B
        self.V = V

        self.state_dimension = B.shape[0]
        self.input_dimension = B.shape[1]

        self.__controller = None

        self.state_cov = None

        self.trajectory = list()
        self.input_trajectory = list()

        self.current_state = None

    @property
    def controller(self):
        return self.__controller

    @controller.setter
    def controller(self, controller):
        assert controller.K.shape == (self.input_dimension, self.state_dimension)

        self.__controller = controller
        self.state_cov = self.state_covariance()

    def reset_system(self, x0):
        assert x0.shape == (self.state_dimension, 1)

        self.trajectory = list()
        self.input_trajectory = list()

        self.current_state = x0
        self.trajectory.append(self.current_state)

    def simulate(self, steps=1):
        assert self.current_state is not None

        start = len(self.trajectory) - 1

        for i in range(0, steps):
            u = self.controller(self.current_state)

            noise = np.random.multivariate_normal(np.zeros(self.state_dimension), self.V, size=1).T
            self.current_state = self.A @ self.current_state + self.B @ u + noise

            self.trajectory.append(self.current_state)
            self.input_trajectory.append(u)

        return np.hstack(self.trajectory[start:]), np.hstack(self.input_trajectory[start:])

    def optimal_controller(self, Q, R):
        assert Q.shape == (self.state_dimension, self.state_dimension)
        assert R.shape == (self.input_dimension, self.input_dimension)

        P = np.array(solve_discrete_are(self.A, self.B, Q, R))
        K_opt = - np.linalg.inv(R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        return K_opt

    # Will not alter internal state, use 'simulate'
    def sample_trajectory(self, x0, length=1):

        assert x0.shape == (self.state_dimension, 1)
        assert self.controller is not None

        x = np.zeros((self.state_dimension, length + 1))
        u = np.zeros((self.input_dimension, length))

        noise = np.random.multivariate_normal(np.zeros(self.state_dimension), self.V, size=length).T

        x[:, [0]] = x0

        for i in range(1, length + 1):
            idx_prev = [i - 1]
            idx = [i]

            u[:, idx_prev] = self.controller(x[:, idx_prev])

            x[:, idx] = self.A @ x[:, idx_prev] + self.B @ u[:, idx_prev] + noise[:, idx_prev]

        ret = {
            'x': x,
            'u': u,
        }
        return ret

    # Will not alter internal state, use 'simulate'
    def sample_trajectories(self, x0s, length=1):

        predictors = list()
        dependents = list()

        for x0 in x0s.T:
            x0 = x0.reshape(-1, 1)
            trajectory = self.sample_trajectory(x0=x0, length=length)

            predictor = np.vstack((trajectory['x'][:, 0:-1], trajectory['u'][:, 0:]))
            dependent = trajectory['x'][:, 1:]

            assert predictor.shape[1] == dependent.shape[1]

            predictors.append(predictor)
            dependents.append(dependent)

        ret = {
            'predictor': np.hstack(predictors),
            'dependent': np.hstack(dependents),
        }

        return ret

    def sample_steady_state(self, n=1):
        return np.random.multivariate_normal(np.zeros(self.state_dimension), self.state_cov, size=n)\
            .reshape(self.state_dimension, n)

    # For non-exciting controllers
    def state_covariance(self):

        assert self.controller is not None
        assert self.controller.K is not None
        assert self.controller.K.shape == (self.input_dimension, self.state_dimension)

        K = self.controller.K
        A_cl = self.A + self.B @ K
        d = self.A.shape

        X_V = pic.SymmetricVariable('X_V', shape=d)

        F = pic.Problem()

        F.set_objective('min', pic.trace(X_V))

        F.add_constraint(A_cl.T * X_V * A_cl - X_V + self.V == 0)
        F.add_constraint(X_V >> 0)

        F.solve(verbosity=0, primals=None)

        # Unstable, so expected variance is infinite
        if F.status != 'optimal':
            return np.ones(d) * np.Inf

        X_V = np.atleast_2d(X_V.value)

        return X_V

class OneDimensionalToy(DiscreteTimeLinearSystem):

    def __init__(self):
        A = np.array([
            [1.01]
        ])

        B = np.array([
            [1]
        ])

        V = np.eye(1) * 1e-3

        super().__init__(A, B, V)

class DoubleIntegrator(DiscreteTimeLinearSystem):

    def __init__(self):
        A = np.array([
            [1, 0.2],
            [0, 1]
        ])

        B = np.array([
            [0],
            [.7]
        ])

        V = np.eye(2) * 1e-3

        super().__init__(A, B, V)


class GraphLaplacian3D(DiscreteTimeLinearSystem):

    def __init__(self):
        A = np.array([
            [1.01, 0.01, 0.00],
            [0.01, 1.01, 0.01],
            [0.00, 0.01, 1.01],
        ])

        B = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])

        V = np.eye(3) * 1e-3

        super().__init__(A, B, V)


if __name__ == "__main__":

    system = DoubleIntegrator()
    system.controller = StateFeedbackLaw(K=system.optimal_controller(Q=np.eye(2), R=np.eye(1)))

    x0s = system.sample_steady_state(3)

    data = system.sample_trajectory(x0s[:, [0]], length=100)

    import matplotlib.pyplot as plt

    plt.plot(data['x'].T)
    plt.plot(data['u'].T)

    plt.show()

    data = system.sample_trajectories(x0s, length=10)
    plt.plot(data['predictor'][0, :].T, label='predictor')
    plt.plot(data['dependent'][0, :].T, label='dependent')
    plt.legend()
    plt.show()