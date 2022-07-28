import numpy as np
import picos as pic
from scipy.linalg import solve_discrete_are

from betl.stability_analysis import check_stability

class LinearQuadraticCostAnalysis:

    def __init__(self, uncertainStateSpaceModel, Q, R):
        self.ussm = uncertainStateSpaceModel
        self.Q = Q
        self.R = R

        self.stab_eps = 1e-6

    # Input variance for a normal distributed excitation input signal added to the control input
    def expected_cost(self, K, episode_length=100, samples=100, c=.95, input_variance=None):

        lqr_costs = list()

        As, Bs, Vs = self.ussm.sample(n=samples, c=c)

        for A, B, V in zip(As, Bs, Vs):

            input_cost = 0
            if input_variance is not None:
                V = V + B @ input_variance @ B.T
                input_cost = episode_length * np.trace(input_variance @ self.R)

            mean_cost = self.quadratic_cost(K, A, B, V, episode_length) + input_cost
            lqr_costs.append(mean_cost)

        return lqr_costs

    # Distribution over the cost if we had perfect model knowledge
    def expected_optimal_cost(self, episode_length=100, samples=100, c=.95):

        lqr_costs = list()

        As, Bs, Vs = self.ussm.sample(n=samples, c=c)

        for A, B, V in zip(As, Bs, Vs):

            P = np.array(np.array(solve_discrete_are(A, B, self.Q, self.R)))
            K = - np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
            mean_cost = self.quadratic_cost(K, A, B, V, episode_length)
            lqr_costs.append(mean_cost)

        return lqr_costs

    def expected_optimality_gap(self, K, episode_length=100, samples=100, c=.95):

        lqr_costs = list()

        As, Bs, Vs = self.ussm.sample(n=samples, c=c)

        for A, B, V in zip(As, Bs, Vs):

            P = np.array(np.array(solve_discrete_are(A, B, self.Q, self.R)))
            K_opt = - np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)

            optimal_cost = self.quadratic_cost(K_opt, A, B, V, episode_length)
            mean_cost = self.quadratic_cost(K, A, B, V, episode_length)
            lqr_costs.append(mean_cost - optimal_cost)

        return lqr_costs

    def quadratic_cost_with_input(self, K, A, B, V, input_variance, episode_length=1):

        V = V + B @ input_variance @ B.T
        input_cost = episode_length * np.trace(input_variance @ self.R)

        mean_cost = self.quadratic_cost(K, A, B, V, episode_length) + input_cost
        return mean_cost

    def quadratic_cost(self, K, A, B, V, episode_length=1):

        A_cl = A + B @ K
        if not check_stability(A_cl, eps=self.stab_eps):
            return np.Inf

        Q_cl = self.Q + K.T @ self.R @ K

        d = A.shape
        X_Q = pic.SymmetricVariable('X_Q', shape=d)

        F = pic.Problem()

        F.set_objective('min', pic.trace(X_Q))

        F.add_constraint(A_cl.T * X_Q * A_cl - X_Q + Q_cl == 0)
        F.add_constraint(X_Q >> 0)

        F.solve(verbosity=0, primals=None)

        # Unstable, so expected cost is infinite
        if F.status != 'optimal':
            return np.Inf

        X_Q = np.atleast_2d(X_Q.value)

        return np.trace(episode_length * (V @ X_Q))

    def optimality_gap(self, K, A, B, V, episode_length=1):
        P = np.array(np.array(solve_discrete_are(A, B, self.Q, self.R)))
        K_opt = - np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)

        optimal_cost = self.quadratic_cost(K_opt, A, B, V, episode_length)
        mean_cost = self.quadratic_cost(K, A, B, V, episode_length)

        return mean_cost - optimal_cost



class EmpiricalQuadraticCostAnalysis:

    def __init__(self, system, Q, R):

        self.system = system
        self.Q = Q
        self.R = R

        # TODO This works for a static reference only. Which is fine for now
        self.x0 = system.current_reference_state
        self.u0 = system.current_reference_input

    def lqr_sum(self, n=100, samples=100):
        cost_sum = list()

        for _ in range(samples):

            offset = 10
            data = self.system.create_trajectory(self.x0, n=n+offset)

            x = data['x']
            u = data['u']

            # import matplotlib.pyplot as plt
            # plt.plot(x[0,:].T)
            # plt.show()

            cost = list()
            for i in range(offset, u.shape[1]):

                x_i = x[:,[i]] - self.x0
                u_i = u[:, [i]] - self.u0

                if self.system.empirically_unstable(x_i + self.x0, u_i + self.u0):
                   return np.Inf, np.Inf

                c_i = x_i.T @ self.Q @ x_i + u_i.T @ self.R @ u_i
                cost.append(c_i)

            cost = np.array(cost)
            cost_sum.append(cost.sum())

        cost_sum = np.array(cost_sum)
        return cost_sum.mean(), cost_sum.std()


if __name__ == "__main__":

    from betl.uncertain_state_space_model import UncertainStateSpaceModel
    from betl.linear_system import GraphLaplacian3D
    from betl.linear_system import NormalRandomControlLaw, StateFeedbackLaw, FeedbackWithNormalRandomControlLaw
    from betl.uncertain_state_space_model import UncertainStateSpaceModel, MatrixNormal

    noise = np.ones((3, 3))*0.01 + np.eye(3) * 0.01

    #noise[2, 2] *= 10

    controller = NormalRandomControlLaw(variance=.1)

    system = GraphLaplacian3D(controller, {'process_noise': noise})
    A = system.A
    B = system.B

    M = np.hstack((A, B)).T
    U = np.eye(6) * 0.01
    V = noise

    dist = MatrixNormal(dim=(6, 3), V0=V*(1000-3-1), v0=1000, L0=np.linalg.inv(U), S0=M)

    ussm = UncertainStateSpaceModel(dist, (3, 3))

    Q = np.eye(3)
    R = np.eye(3) * 1.

    input_noise_ = .2
    input_noise = np.ones(system.input_dimension) * input_noise_
    input_noise = np.diag(input_noise)

    K = system.optimal_controller(Q, R)
    K = np.array([[-1.56910583e+00, -2.19443081e-01, -6.42026512e-03],
       [ 1.12376636e-03, -1.39334249e+00, -4.57225448e-01],
       [ 1.91300239e-01, -3.35152178e-01, -1.34792092e+00]])
    controller = FeedbackWithNormalRandomControlLaw(K=K,
                                                    variance=input_noise)
    system.controller = controller

    n = 1
    emp = EmpiricalQuadraticCostAnalysis(system, Q, R)
    cost_emp, cost_emp_v = emp.lqr_sum(n=n, samples=200)
    print('Emp. cost for the given system {0}+-{1}'.format(cost_emp, cost_emp_v))

    test = LinearQuadraticCostAnalysis(uncertainStateSpaceModel=ussm, K=controller.K, Q=Q, R=R)
    # cost_lmi = test.sample_cost(test.K, system.A, system.B, V=noise, n=n)

    B = system.B
    A, B, V = ussm.sample(1)
    A, B, V = A[0], B[0], V[0]
    V = V + B @ input_noise @ B.T
    cost_lmi = test.sample_cost(K=controller.K, A=A, B=B, V=V, n=n) + n * np.trace(input_noise @ R)

    print('Anl. cost for the given system {0}'.format(cost_lmi))

    #cost_exp = test.expected_cost(n, samples=500)

    #print('Exp. cost over uncertain system {0}+-{1}'.format(np.ma.masked_invalid(cost_exp).mean(),
    #                                                        np.ma.masked_invalid(cost_exp).std()))
