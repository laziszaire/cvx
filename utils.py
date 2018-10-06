__author__ = "Li Tao, ltipchrome@gmail.com"


def backtrack(x_cur, f, grad, descent_derection, alpha=.3, beta=.5):
    """
    backtracking line search

    :param x_cur:
    :param f:
    :param grad:
    :param descent_derection:
    :param alpha:
    :param beta:
    :return: learning rate, t
    """
    t = 1
    _f = f(x_cur)
    _grad = grad(x_cur)
    while f(x_cur + t*descent_derection) > (_f + t*alpha*_grad*descent_derection):
        t *= beta
    return t
