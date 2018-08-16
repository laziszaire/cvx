# equality constraints
from unconstrained import line_search
from numpy.linalg import inv, solve, norm
import numpy as np
import copy


class EqualityCvx:
    def __init__(self, f0, grad_f, hessian_f, A, b, x0, epsilon=1e-5, alpha=.3, beta=.8):
        self.f0 = f0
        self.A = A
        self.b = b
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.x0 = x0
        self.grad = grad_f
        self.hessian = hessian_f
        self._hessian = hessian_f(x0)
        self._grad = grad_f(x0)
        self.x = x0
        self.n_iter = 0
        self._decrement = None
        self.optimal = None
        self.step = None
        self.w = np.random.randn(A.shape[0])
        self.v = np.random.randn(A.shape[0])  # next dual variable can be calculated directly by v_plus = w
        self.v_plus = self.w
        self.dual_step = None

    def newton_step(self, feasible=True):
        """
        solve the KKT system
        """
        x = self.x
        _hessian = self._hessian = self.hessian(x)
        _grad = self._grad = self.grad(x)
        A, b= self.A, self.b
        n, p = A.shape

        first_col = np.vstack([_hessian, A])
        _zero = np.zeros((n, n))
        second_col = np.vstack([A.T, _zero])
        KKT_matrix = np.hstack([first_col, second_col])

        if feasible:
            RHS = np.concatenate((-_grad, np.zeros(n)))[:, np.newaxis]
        else:
            RHS = np.concatenate((-_grad, A @ x - b))[:, np.newaxis]
        sol = solve(KKT_matrix, RHS)
        step = sol.flatten()[:p]
        w = sol.flatten()[-n:]
        self.dual_step = w - self.w
        self.step = step
        self.w = w
        self.v_plus = w
        return step

    @property
    def newton_decrement(self):
        step, hessian = self.step, self._hessian
        self._decrement = (step.T@hessian@step)**(1/2)
        return self._decrement

    def newton_method(self):
        while True:
            epsilon, f0, alpha, beta = self.epsilon, self.f0, self.alpha, self.beta
            step = self.newton_step(feasible=True)
            decrement = self.newton_decrement
            print(decrement)
            if 1/2 * decrement**2 < epsilon:
                break
            t = line_search(step, self.x, f0, self._grad, alpha=alpha, beta=beta)
            self.x += t*step
            self.n_iter += 1
        self.optimal = self.x

    def infeasible_newton(self):
        epsilon, alpha, beta, A, b = self.epsilon, self.alpha, self.beta, self.A, self.b
        while True:
            _ = self.newton_step(feasible=False)
            t = self.infeasible_line_search()
            nr = norm(self.residual(self.x, self.v))
            if np.isclose(A@self.x, b).all() and (nr <= epsilon):
                break
            self.x += t * self.step
            self.v = self.w
        self.optimal = self.x

    def residual(self, x, v):
        A, b, _grad = self.A, self.b, self.grad(x)
        r = np.concatenate([_grad + A.T@v, A@x - b])
        return r

    def infeasible_line_search(self):
        t = 1
        alpha, beta, x, v = self.alpha, self.beta, self.x, self.v
        nr_before = norm(self.residual(x, v))
        while True:
            print('line search')
            nr = norm(self.residual(x + t*self.step, v + t*self.dual_step))
            if nr < (1-alpha*t)*nr_before:
                break
            t = beta * t
        return t

    def minimize(self, feasible=True):
        if feasible:
            self.newton_method()
        else:
            self.infeasible_newton()
        return self.optimal

# ------------------------------ interior-point method -----------------------
# x*(t) is m/t-suboptimal


def log_barrier(u, t):
    return -(1/t)*np.log(u)


class IneCvx(EqualityCvx):

    def __init__(self, f0, fis, grad_f, hessian_f, A, b, x0, epsilon=1e-5, alpha=.3, beta=.8, u=20):
        super().__init__(f0, grad_f, hessian_f, A, b, x0, epsilon=epsilon, alpha=alpha, beta=beta)
        self.fis = fis
        self.u = u
        self.ecvx = self.copy()

    def copy(self):
        return copy.deepcopy(self)

    def barrier_method(self):
        # centering step
        t = self.choose_t()
        fis = self.fis
        m = len(fis)
        while True:
            # outer loop
            self.ecvx = self.copy()
            self.ecvx.f0 = lambda x: t * (self.f0(x) + sum(log_barrier(_fi(x), t) for _fi in fis))
            self.ecvx.grad = self.grad_ft(t)
            self.ecvx.hessian = self.hessian_ft(t)
            self.ecvx.x0 = self.x
            self.x = self.ecvx.minimize()
            if m/t < self.epsilon:
                break
            t *= self.u
        self.optimal = self.x
        return self.x

    def choose_t(self):
        return NotImplementedError

    def grad_ft(self, t):
        return NotImplementedError

    def hessian_ft(self, t):
        return NotImplementedError


def test_ecvx():
    # equality constrained analytic centering problem
    # n = 3
    # m = 2
    A = np.random.randn(2, 3)
    x0 = np.abs(np.random.randn(3))
    b = A@x0
    f0 = lambda x: -np.sum(np.log(x))
    grad_f = lambda x: np.asarray(-1/x)
    hessian_f = lambda x: np.diag(1/x**2)
    ecvx = EqualityCvx(f0, grad_f, hessian_f, A, b, x0)
    x = ecvx.minimize()
    return ecvx


def test_ecvx_feasible():
    A = np.random.randn(2, 3)
    x0 = np.ones(3)
    b = A@x0
    f0 = lambda x: -np.sum(np.log(x))
    grad_f = lambda x: np.asarray(-1/x)
    hessian_f = lambda x: np.diag(1/x**2)
    ecvx = EqualityCvx(f0, grad_f, hessian_f, A, b, x0)
    x = ecvx.minimize(feasible=False)
    A = ecvx.A
    dual = -np.sum(np.log(1 / (A.T @ ecvx.w))) + ecvx.w @ (A @ (1 / (A.T @ ecvx.w)) - ecvx.b)
    assert (f0(x) - dual) <= 1e-5


if __name__ == "__main__":
    a = test_ecvx_feasible()



















