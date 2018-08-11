import numpy as np
from numpy.linalg import inv, norm


def line_search(step, x, f, grad, alpha=.3, beta=.8):
    """
    Backtracking line search
    linear extrapolation estimation:
        f(x+t*step) = f(x) + t*grad(x)*step
        estimated_decrement = f(x) - f(x+t*step)
                            = t*grad(x)*step
        alpha_estimated_decrement = alpha*estimated_decrement
                                  = alpha*t*grad(x)*step
        if real_decrement > alpha_estimated_decrement
            accept t
    alpha: typically chosen between .01~.3
    beta between .1~.8
    """
    t = 1
    while _backtrack_continue(x, step, t, f, grad, alpha=alpha):
        t = beta*t
    return t


def _backtrack_continue(x, step, t, f, grad, alpha=.3):
    """
    stop criterion of backtracking
    :param x:
    :param step: descent direction
    :param t:
    :param alpha:
    :param f: object function
    :param grad:  gradient function
    :return:
    """
    return f(x + t*step) > f(x) + alpha*t*grad.dot(step)


def descent(x0, f, grad, step, alpha=.3, beta=.8):
    """
    general descent method
    :param x0:
    :param step:
    :param f:
    :param grad:
    :param alpha:
    :param beta:
    :return:
    """
    x = x0
    while True:
        descent_step = step(x)
        _grad = grad(x)
        t = line_search(descent_step, x, f, _grad, alpha=alpha, beta=beta)
        x1 = x + t*descent_step
        if check_stop(x1, x):
            break
    return x1


def check_stop(*args, **kwargs):
    """
    check stop
    """
    return NotImplementedError


def gradient_descent(x0, f, grad, eta=1e-5, alpha=.3, beta=.8):
    x = x0
    while True:
        _grad = grad(x)
        step = -_grad
        if norm(step) <= eta:
            break
        t = line_search(step, x, f, _grad, alpha=alpha, beta=beta)
        x += t*step
    return x


def steepest_step(grad, norm='L2'):
    """
    steepest_step = argmin{grad.dot(v) | norm(v) <= 1}
    :param grad:
    :param norm:
    :return:
    """
    if norm == 'L2':
        return -grad
    else:
        return NotImplementedError


def steepest_descent(x0, f, grad, eta=1e-5, alpha=.3, beta=.8):
    x = x0
    while True:
        _grad = grad(x)
        _steepest_step = steepest_step(_grad)
        if norm(_steepest_step) <= eta:
            break
        t = line_search(_steepest_step, x, f, _grad, alpha=alpha, beta=beta)
        x += t*_steepest_step
    return x


def newton_step(hessian, grad):
    hessian_inverse = inv(hessian)
    return -hessian_inverse@grad


def newton_decrement(hessian, grad):
    inv_hessian = inv(hessian)
    # _decrement = (newton_step.T@hessain@newton_step)**.5
    _decrement = (grad@inv_hessian@grad)**(1/2)
    return _decrement


def newton_method(f, grad, hessian, x0, epsilon=1e-5, alpha=.3, beta=.8):
    x = x0
    while True:
        _grad = grad(x)
        _hessian = hessian(x)
        inv_hessian = inv(_hessian)
        new_step = -inv_hessian@_grad
        lambda_square = _grad.T@inv_hessian@_grad
        if lambda_square/2 <= epsilon:
            break
        t = line_search(new_step, x,  f, _grad, alpha=alpha, beta=beta)
        x += t*new_step
    return x


f0 = lambda x: .5 * x ** 2
grad_f0 = lambda x: np.asarray([x])
hessian_f0 = lambda x: np.asarray([[1]])
x0 = np.asscalar(np.random.randn(1) * 100)


def test_newton():
    x = newton_method(f0, grad_f0, hessian_f0, x0)
    assert x < 1e-5


def test_gradient():
    x = gradient_descent(x0, f0, grad_f0,)
    assert x < 1e-5


def test_steepest():
    x = steepest_descent(x0, f0, grad_f0)
    assert x < 1e-5


def sanity_check():
    test_gradient()
    test_steepest()
    test_newton()


if __name__ == "__main__":
    sanity_check()


