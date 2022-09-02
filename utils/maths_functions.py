from numpy import cosh, sinh, exp, real, pi, abs, max, broadcast_to, log
import numpy as np
from scipy.special import gamma as gammafnc
from scipy.special import hankel1, hankel2, gammainc, gammaincc
from scipy.stats import levy_stable
from scipy.special import gammaincinv
from scipy.stats import gamma as gammaDist

def incgammau(s, x):
    return gammaincc(s, x) * gammafnc(s)


def incgammal(s, x):
    return gammainc(s, x) * gammafnc(s)


def psi(x, alpha, lambd):
    return -alpha * (cosh(x) - 1) - lambd * (exp(x) - x - 1)


def dpsi(x, alpha, lambd):
    return -alpha * sinh(x) - lambd * (exp(x) - 1)


def hankel_squared(lam, z):
    return real(hankel1(lam, z) * hankel2(lam, z))


def get_z0_H0(lambd):
    a = pi * (1.0 - 2.0 * abs(lambd)) ** 2
    b = gammafnc(np.abs(lambd)) ** 2
    c = 1 / (1 - 2 * np.abs(lambd))
    z1 = (a / b) ** c
    H0 = z1 * hankel_squared(np.abs(lambd), z1)
    return z1, H0


def logsumexp(w, h, x, axis=0, retlog=False):
    c = np.max(w)
    broad_l = broadcast_to((w - c).flatten(), x.T.shape).T
    if retlog:
        return c + log((exp(broad_l) * h(x)).sum(axis=axis))
    return exp(c) * (exp(broad_l) * h(x)).sum(axis=axis)
