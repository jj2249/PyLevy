from scipy.special import gammainc, gammainc, gammaincinv
from scipy.special import gamma as gammafnc


def incgammau(s, x):
	return gammaincc(s, x)*gammafnc(s)


def incgammal(s, x):
	return gammainc(s, x)*gammafnc(s)