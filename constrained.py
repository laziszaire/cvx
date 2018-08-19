# equality constraints
from unconstrained import line_search
from numpy.linalg import inv, solve, norm
import numpy as np
import copy
from unconstrained import gradient_descent


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
        self.dual = None
        self.duality_gap = None

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
            RHS = -np.concatenate((_grad, A@x - b))[:, np.newaxis]
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
    return -(1/t)*np.log(-u)


class IneCvx(EqualityCvx):

    def __init__(self, f0, fis, grads, hessians, A, b, x0, dual=None, epsilon=1e-5, alpha=.3, beta=.8, u=20):
        super().__init__(f0, grads[0], hessians[0], A, b, x0, epsilon=epsilon, alpha=alpha, beta=beta)
        self.fis = fis
        self.u = u
        self.grad_fis = grads[1:]
        self.hessian_fis = hessians[1:]
        self.ecvx = EqualityCvx(f0, grads[0], hessians[0], A, b, x0)
        self.dual = dual

    def copy(self):
        return copy.deepcopy(self)

    def minimize(self, feasible=True):
        x = self.barrier_method()
        return x

    def barrier_method(self):
        # centering step
        t = self.init_t()
        fis = self.fis
        m = len(fis)
        while True:
            # outer loop
            self.ecvx.f0 = lambda x: t*self.f0(x) + sum(log_barrier(_fi(x), 1) for _fi in fis)  # 细心
            self.ecvx.grad = self.grad_ft(t)
            self.ecvx.hessian = self.hessian_ft(t)
            self.ecvx.x0 = self.x
            self.x = self.ecvx.minimize()
            self.duality_gap = m/t  # lambda = -1/(t*fi(x)), fi(x)<=0 ==> lambda >=0
            if m/t < self.epsilon:
                break
            t *= self.u
            print(f"duality gap is {self.duality_gap}")
        self.optimal = self.x
        return self.x

    def grad_ft(self, t):
        # grad of logarithm barrier = sum(-grad_fi/fi)
        grad_barrier = lambda x: sum(-gi(x)/fi(x) for fi, gi in zip(self.fis, self.grad_fis))
        _grad = lambda x: t*self.grad(x) + grad_barrier(x)
        return _grad

    def hessian_ft(self, t):
        x_xt = lambda x: x[:, np.newaxis]@x[:, np.newaxis].T
        hessian_barrier = lambda x: sum(x_xt(gi(x))/(fi(x)**2) - hi(x)/fi(x)
                                        for fi, gi, hi in zip(self.fis, self.grad_fis, self.hessian_fis))
        _hessian = lambda x: t*self.hessian(x) + hessian_barrier(x)
        return _hessian

    def init_t(self, method=None):
        # todo test
        if method == 'dual':
            m = len(self.fis)
            v = np.random.randn(self.A.shape[0])
            _lambda = np.asarray([1])
            eita = self.f0(self.x0) - self.dual(_lambda, v)
            t0 = m/eita
        elif method == 'central_path':
            # minimizer residual norm(t*grad)
            residual = lambda t, grad, grad_barrier, A, v: t*grad + grad_barrier + A.T@v
            grad_barrier = lambda x: sum(-gi(x) / fi(x) for fi, gi in zip(self.fis, self.grad_fis))
            f0 = lambda x: residual(x[0], self.grad(self.x0), grad_barrier(self.x0), self.A, x[1:])
            _grad = lambda x: np.concatenate(2*f0(x).dot(self.grad(self.x0)),
                                             2*f0(x)@self.A.T)
            x0 = np.random.randn(self.A.shape[0]+1)
            x = gradient_descent(x0, f0, _grad)
            t0 = x[0]
        elif method == 'hessian_central':
            return NotImplementedError
        elif method == 'epsilon':
            m = len(self.fis)
            t0 = m/self.epsilon * 100
        elif method is None:
            t0 = 10
        return t0


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


def test_ecvx_infeasible():
    """
    test infeasible start Newton method equality constrained problem
    :return:
    """
    A = np.random.randn(2, 3)
    x0 = np.ones(3)
    b = A@np.abs(np.random.randn(3))
    f0 = lambda x: -np.sum(np.log(x))
    grad_f = lambda x: np.asarray(-1/x)
    hessian_f = lambda x: np.diag(1/x**2)
    ecvx = EqualityCvx(f0, grad_f, hessian_f, A, b, x0)
    x = ecvx.minimize(feasible=False)
    A = ecvx.A
    dual = -np.sum(np.log(1 / (A.T @ ecvx.w))) + ecvx.w @ (A @ (1 / (A.T @ ecvx.w)) - ecvx.b)
    print(f'dual grap is {f0(x) - dual}')
    assert (f0(x) - dual) <= 1e-5
    return ecvx


def test_barrier():
    A = np.random.randn(2, 3)
    x0 = np.abs(np.random.randn(3))
    b = A @ x0
    f0 = lambda x: -np.sum(np.log(x))
    grad_f = lambda x: np.asarray(-1 / x)
    hessian_f = lambda x: np.diag(1 / x ** 2)
    # sum(x) <= 1
    fis = (lambda x: x.dot(np.ones_like(x)) - 100,)
    # todo: dual
    grads = (grad_f, lambda x: np.ones_like(x))
    hessians = (hessian_f, lambda x: np.zeros((x.shape[0], x.shape[0])))
    iecvx = IneCvx(f0, fis, grads, hessians, A, b, x0)
    _ = iecvx.minimize()
    return iecvx


if __name__ == "__main__":
    a = test_barrier()
    print(a)