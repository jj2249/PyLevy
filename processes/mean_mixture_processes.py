import numpy as np
from PyLevy.processes import base_processes

class __MeanMixtureLevyProcess(base_processes.__LevyProcess):
	def __init__(self, mu, mu_W, var_W, subordinator, rng=np.random.default_rng()):
		self.__rng = rng
		self.__mu = mu
		self.__mu_W = mu_W
		self.__var_W = var_W
		self.__std_W = np.sqrt(var_W)
		self.__subordinator = subordinator


	def simulate_path(self, observation_times):
		subordinator_sample = self.__subordinator.simulate_jumps()
		
		mean_mix_jtimes = subordinator_sample[0]
		mean_mix_jsizes = self.__mu_W*subordinator_sample[1] + self.__std_W*np.sqrt(subordinator_sample[1])*self.__rng.normal(size=subordinator_sample[1].shape)

		integrated_process = self.integrate(observation_times, mean_mix_jtimes, mean_mix_jsizes, drift=self.__mu)

		return integrated_process

class NormalTemperedStableProcess(__MeanMixtureLevyProcess):

	def __init__(self, alpha, beta, C, mu, mu_W, var_W, rng=np.random.default_rng()):
		self.__tsp = base_processes.TemperedStableProcess(alpha=alpha, beta=beta, C=C, rng=rng)
		super().__init__(mu, mu_W, var_W, self.__tsp, rng=rng)


class NormalGammaProcess(__MeanMixtureLevyProcess):

	def __init__(self, beta, C, mu, mu_W, var_W, rng=np.random.default_rng()):
		self.__gp = base_processes.GammaProcess(beta=beta, C=C, rng=rng)
		super().__init__(mu, mu_W, var_W, self.__gp, rng=rng)


class GeneralHyperbolicProcess(__MeanMixtureLevyProcess):
	# using GIG process
	def __init__(self):
		pass
