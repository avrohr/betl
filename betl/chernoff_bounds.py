import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import matrix_power
from scipy.optimize import minimize

import picos as pic

def state_covariance(A, B, K, V):

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


# E is the steady state state covariance
def bounds(E, Q, R, A, B, K, n, p):

    Q_ = Q + K.T @ R @ K
    A_ = A + B @ K

    omega = block_diag(*list(Q_ for i in range(n)))

    column_list = list()
    for i in range(n):
        entry_list = list()
        for j in range(n):
            exp = np.abs(j-i)
            if j >= i:
                entry = matrix_power(A_, exp) @ E
            else:
                entry = E @ matrix_power(A_, exp).T
            entry_list.append(entry)

        column = np.vstack(entry_list)
        column_list.append(column)

    cov = np.hstack(column_list)
    assert np.allclose(cov, cov.T)

    M = omega @ cov
    eig = np.linalg.eigvals(M + np.eye(M.shape[0]) * 1e-9)

    assert 0 < p < 1
    beta = p
    # assert np.alltrue(0 < eig < 1)
    def x(eta):
        return - 1/eta * np.log(beta/2) - 1/(2*eta) * np.sum(np.log(1 - 2 * eta * eig))

    test = np.linspace(-1000., 0., 10000, endpoint=False)
    f = lambda eta: -x(eta)
    xs = list(f(eta) for eta in test)

    # import matplotlib.pyplot as plt
    # plt.plot(test, xs)
    # plt.show()
    # lower Bound
    bnds = ((None, -1e-6),)
    res = minimize(fun=lambda eta: -x(eta), x0=test[np.argmin(xs)], bounds=bnds)
    k_m = x(res.x)


    max = 1/(2 * np.max(eig))

    test = np.linspace(0.0001, max, 1000, endpoint=False)
    f = lambda eta: x(eta)
    xs = list(f(eta) for eta in test)

    bnds = ((1e-4, max),)
    res = minimize(x, x0=test[np.argmin(xs)], bounds=bnds)
    k_p = x(res.x)

    return k_m, k_p



# A = np.random.rand(3, 3)
# B = np.random.rand(3, 2)
#
# Q = np.eye(3)
# R = np.eye(3) * 10
#
# from scipy.linalg import solve_discrete_are
#
# P = np.array(np.array(solve_discrete_are(A, B, Q, R)))
# K = - np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
#
#
# bounds(np.eye(3) * 0.01, Q, R, A, B, K, 3)
# A = np.array([[1.01, 0.01, 0.  ],
#        [0.01, 1.01, 0.01],
#        [0.  , 0.01, 1.01]])
#
# B = np.array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])
#
# Q = np.eye(3)
# R = np.eye(3) * 10
# cov = np.array([[ 0.09312158, -0.00558783,  0.03694939],
#        [-0.00558783,  0.0911873 , -0.00599917],
#        [ 0.03694939, -0.00599917,  0.1346676 ]])
#
# K = np.array([[-0.3364908 , -0.02618322, -0.00035088],
#        [-0.0383956 , -0.33996468, -0.01508168],
#        [ 0.01147953, -0.01979958, -0.33529024]])
#
# V = np.array([[0.05, 0.  , 0.  ],
#        [0.  , 0.05, 0.  ],
#        [0.  , 0.  , 0.05]])
#
# cov_ = state_covariance(A, B, K, V)
# print(bounds(cov_, Q, R, A, B, K, 10, 0.01))