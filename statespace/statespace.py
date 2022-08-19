import numpy as np

class LinearSDEStateSpace:

	def __init__(self, initial_state, model_drift, model_mean, model_covar, model_noise_matrix, driving_process, observation_matrix, observation_noise, rng=np.random.default_rng()):
		"""
		- model drift = e^A(t-u)
		- model mean = e^A(t-u) h
		- model covar = e^A(t-u) h hT e^A(t-u)T
		- model noise matrix = B s.t. Xt = drift @ Xs + B @ noise
		"""
		# need to place checks on the dimensions of the return values of these functors
		self.__state = initial_state
		self.__drift = model_drift
		self.__mean = model_mean
		self.__covar = model_covar
		self.__B = model_noise_matrix

		self.__driving = driving_process
		self.__mu_W = self.__driving.get_mu_W()
		self.__var_W = self.__driving.get_var_W()

		self.__H = observation_matrix
		self.__obs_noise = observation_noise

		self.__rng = rng


	def __increment_state(self, interval, M=100, gamma_0=0.):
		jump_times, jump_sizes = self.__driving.simulate_jumps(rate=1./interval, M=M, gamma_0=gamma_0)
		m_vec = np.atleast_2d((self.__mu_W * jump_sizes * self.__mean(interval, jump_times)).sum(axis=-1)).T
		S_mat = (self.__var_W * jump_sizes * self.__covar(interval, jump_times)).sum(axis=-1)
		try:
			C_mat = np.linalg.cholesky(S_mat)
			e = C_mat @ self.__rng.normal(size=S_mat.shape[1])
		except np.linalg.LinAlgError:
			e = np.zeros(S_mat.shape[1])

		e = np.atleast_2d(e).T
		new_state = self.__drift(interval) @ self.__state + self.__B @ e
		return new_state


	def __observe_in_noise(self):
		return (self.__H @ self.__state + np.sqrt(self.__var_W*self.__obs_noise)*self.__rng.normal()).item()


	def generate_observations(self, times):
		intervals = np.diff(times)
		observations = [self.__observe_in_noise()]
		for diff in intervals:
			self.__state = self.__increment_state(diff)
			observations.append(self.__observe_in_noise())
		return observations


class LangevinStateSpace(LinearSDEStateSpace):

	def __init__(self, initial_state, theta, driving_process, observation_matrix, observation_noise, rng=np.random.default_rng()):
		self.__theta = theta
		super().__init__(initial_state, self.langevin_drift, self.langevin_mean, self.langevin_covar, self.langevin_noise_matrix, driving_process, observation_matrix, observation_noise, rng=rng)

	# model specific functors
	langevin_drift = lambda self, interval : np.exp(self.__theta*interval)*np.array([[0.,1./self.__theta],[0.,1.]]) + np.array([[1.,-1./self.__theta],[0.,0.]])
	langevin_mean = lambda self, interval, jtime : np.exp(self.__theta*(interval-jtime))*np.atleast_2d(np.array([[1./self.__theta, 1]])).T + np.atleast_2d(np.array([[-1./self.__theta, 0.]])).T
	langevin_covar = lambda self, interval, jtime : np.exp(2.*self.__theta*(interval-jtime))*np.array([[1./(self.__theta**2),1./self.__theta],[1./self.__theta,1.]])[:,:,np.newaxis] + np.exp(self.__theta*(interval-jtime))*np.array([[-2./(self.__theta**2),-1./self.__theta],[-1./self.__theta,0]])[:,:,np.newaxis] + np.array([[1./(self.__theta**2),0.],[0.,0.]])[:,:,np.newaxis]
	langevin_noise_matrix = np.eye(2)




