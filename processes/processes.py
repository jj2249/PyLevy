import numpy as np
from PyLevy.utils import maths_functions as mathsf

class __LevyProcess:
	@staticmethod
	def integrate(evaluation_points, t_series, x_series):
		W = [x_series[t_series<point].sum() for point in evaluation_points]
		return np.array(W).T


class __JumpLevyProcess(__LevyProcess):
	def __init__(self, rng=np.random.default_rng()):
		self.rng = rng


	def accept_reject_simulation(self, h_func, thinning_func, rate=1.0, M=100, gamma_0=0.0):
		"""

		"""
		epoch_seq = self.rng.exponential(scale=rate, size=M)
		epoch_seq[0] += gamma_0
		epoch_seq = epoch_seq.cumsum()

		x_seq = h_func(epoch_seq)
		acceptance_seq = thinning_func(x_seq)
		u = self.rng.uniform(low=0.0, high=1.0, size=x_seq.size)
		x_seq = x_seq[u < acceptance_seq]
		times = self.rng.uniform(low=0.0, high=1.0, size=x_seq.size)
		return times, x_seq


class GammaProcess(__JumpLevyProcess):

	def __init__(self, beta=None, C=None, rng=np.random.default_rng()):
		self.set_parameters(beta, C)
		super().__init__(rng=rng)


	def set_parameters(self, beta, C):
		self.beta = beta
		self.C = C


	def get_parameters(self):
		return {"beta" : self.beta, "C" : self.C}


	def h_func(self, epoch):
		return 1./(self.beta*(np.exp(epoch/self.C)-1.))


	def thinning_func(self, x):
		return (1.+self.beta*x)*np.exp(-self.beta*x)


	def simulate_jumps(self, rate=1.0, M=100, gamma_0=0.0):
		return super().accept_reject_simulation(self.h_func, self.thinning_func, rate, M, gamma_0)


	def unit_expected_residual_gamma(self, c):
		return (self.C/self.beta)*mathsf.incgammal(1., 1./(np.exp(c/self.C)-1.))


	def unit_variance_residual_gamma(self, c):
		return (self.C/self.beta**2)*mathsf.incgammal(2., 1./(np.exp(c/self.C)-1.))


class TemperedStableProcess(__JumpLevyProcess):

	def __init__(self, alpha=None, beta=None, C=None, rng=np.random.default_rng()):
		self.set_parameters(alpha, beta, C)
		super().__init__(rng=rng)


	def set_parameters(self, alpha, beta, C):
		self.alpha = alpha
		self.beta = beta
		self.C = C


	def get_parameters(self):
		return {"alpha" : self.alpha, "beta" : self.beta, "C" : self.C}


	def h_func(self, epoch):
		return np.power((self.alpha/self.C)*epoch, np.divide(-1., self.alpha))


	def thinning_func(self, x):
		return np.exp(-self.beta*x)


	def simulate_jumps(self, rate=1.0, M=100, gamma_0=0.0):
		return super().accept_reject_simulation(self.h_func, self.thinning_func, rate, M, gamma_0)


	def unit_expected_residual_tempered_stable(self, c):
		return (self.C*self.beta**(self.alpha-1.))*mathsf.incgammal(1.-self.alpha, self.beta*(self.alpha*c/self.C)**(-1./self.alpha))


	def unit_variance_residual_tempered_stable(self, c):
		return (self.C*self.beta**(self.alpha-2.))*mathfs.incgammal(2.-self.alpha, self.beta*(self.alpha*c/self.C)**(-1./self.alpha))