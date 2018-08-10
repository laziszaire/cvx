import numpy as np
from numpy.linalg import inv

def backtrack(delta, x, alpha, beta, f, grad):
    """
    Backtracking line search
    linear extraplation estimation:
        f(x+t*delta) = f(x) + t*grad(x)*delta
        estimated_decrement = f(x) - f(x+t*delta)
                            = t*grad(x)*delta
        alpha_estimated_decrement = alpha*estimated_decrement
                                  = alpha*t*grad(x)*delta
        if real_decrement > alpha_estimated_decrement
            accept t
    """
    t = 1
    while _backtrack_continue(x, delta, t, alpha, f, grad) :
        t = beta*t
    return t


def _backtrack_continue(x, delta, t, alpha, f, grad):
    """
    stop criterion of backtracking
    :param x:
    :param delta: descent direction
    :param t:
    :param alpha:
    :param f: object function
    :param grad:  gradient function
    :return:
    """
    return f(x + t * delta) > f(x) + alpha*t*grad(x).dot(delta)


def descent(x0, delta, f, grad, alpha, beta):
    x = x0
    _stop = False
    while not _stop:
        t = backtrack(delta(x), x, alpha, beta, f, grad)
        x1 = x + t*delta(x)
        _stop = check_stop(x1, x)
        x = x1
    return


def check_stop(x1, x):
    """
    check stop
    """
    return NotImplementedError

def newton_step(hessian, grad):
    hessian_inverse = inv(hessian)
    return -hessian_inverse@grad


def newton_decrement(hessian, grad):
    inv_hessian = inv(hessian)
    # _decrement = (newton_step.T@hessain@newton_step)**.5
    _decrement = (grad@inv_hessian@grad)**.5
    return _decrement

def newton_method(f, grad, hessian, x0, epsilon,alpha, beta):
    x = x0
    stop = False
    while not stop:
        _grad = grad(x)
        _hessian = hessian(x)
        inv_hessian = inv(_hessian)
        new_step = -inv_hessian@_grad
        lambda2 = _grad.T@inv_hessian@_grad
        stop = lambda2 / 2 <= epsilon
        t = backtrack(new_step, x, alpha, beta, f, grad)
        x += t*new_step
    return





