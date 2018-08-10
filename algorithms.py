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
    return x + t*delta


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
