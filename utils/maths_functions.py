from numpy import cosh, sinh, exp, real
import numpy as np
from scipy.special import gamma as gammafnc
from scipy.special import hankel1, hankel2, gammainc, gammaincc


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


def logsumexp(w, h, x, axis=0, retlog=False):
	c = np.max(w)
	broad_l = np.broadcast_to((w-c).flatten(), x.T.shape).T
	if retlog:
		return c + np.log((np.exp(broad_l) * h(x)).sum(axis=axis))
	return np.exp(c)*(np.exp(broad_l) * h(x)).sum(axis=axis)
