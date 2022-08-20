from scipy.special import gammainc, gammainc, gammaincinv
from scipy.special import gamma as gammafnc


def incgammau(s, x):
	return gammaincc(s, x)*gammafnc(s)


def incgammal(s, x):
	return gammainc(s, x)*gammafnc(s)


def logsumexp(w, h, x, axis=0, retlog=False):
	c = np.max(w)
	broad_l = np.broadcast_to((w-c).flatten, x.T.shape).T
	if retlog:
		return c + np.log((np.exp(broad_l) * h(x)).sum(axis=axis))
	return np.exp(c)*(np.exp(broad_l) * h(x)).sum(axis=axis)