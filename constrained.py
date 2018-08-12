# equality constraints
from unconstrained import line_search
from numpy.linalg import inv, solve, norm
import numpy as np


def newton_step(grad, hessian, A, x, b, feasible=True):
    """
    solve the KKT system

    :param grad:
    :param hessian:
    :param A:
    :return:
    """

    first_col = np.vstack([hessian, A])
    n, p = A.shape[0]
    _zero = np.zeros((n, n))
    second_col = np.vstack([A.T, _zero])
    KKT_matrix = np.hstack([first_col, second_col])
    if feasible:
        RHS = np.concatenate((-grad, np.zeros(n)))[:, np.newaxis]
    else:
        RHS = np.concatenate((-grad, A@x - b))[:, np.newaxis]
    sol = solve(KKT_matrix, RHS)
    step = sol.flatten()[:n]
    w = sol.flatten()[n:]
    return step, w


def newton_decrement(grad, hessian):
    return (grad.T@inv(hessian)@grad)**(1/2)


def newton_method(f, grad, hessian, A, x0, epsilon=1e-5, alpha=.3, beta=.8):
    x = x0
    while True:
        _grad = grad(x)
        _hessian = hessian(x)
        step, w = newton_step(_grad, _hessian, A)
        decrement = newton_decrement(_grad, _hessian)
        if 1/2 * decrement**2 < epsilon:
            break
        t = line_search(step, f, _grad, alpha=alpha, beta=beta)
        x += t*step
    return x


# infeasible start

def infeasible_newton(grad, hessain, A, x0, b, epsilon=1e-5, alpha=.3, beta=.8):
    x = x0
    while True:
        step, v = newton_step(grad, hessain, A, x, b, feasible=False)
        t = infeasible_line_search(x, v, step, v, alpha=alpha, beta=beta)
        x += t
        if np.isclose(A@x, b) and (residual(x, v) <= epsilon):
            break
    return x


def infeasible_line_search(x, v, primal_step, dual_step, alpha=.3, beta=.8):
    t = 1
    norm_r = lambda x, y: norm(residual(x, y))
    while norm_r(primal_step, dual_step) > (1-alpha*t)*norm_r(x, v):
        t = beta * t
    return t


def residual(x, y):
    pass


# interior-point method
# x*(t) is m/t-suboptimal
def log_barrier(u, t):
    return -(1/t)*np.log(u)


def barrier_method(x0, f, gradt, hessiant, A, fi, epsilon=1e-5, alpha=.3, beta=.8, u=20):
    # centering step
    x = x0
    t = choose_t()
    m = len(fi)
    while True:
        _f = lambda x: t*(f(x) + sum(log_barrier(_fi(x), t) for _fi in fi))
        x = newton_method(_f, gradt(t), hessiant(t), A, x, epsilon=epsilon, alpha=alpha, beta=beta, u=u)
        if m/t < epsilon:
            break
        t *= u
    return x


def choose_t():
    return NotImplementedError



















