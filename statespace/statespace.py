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
		self.state = initial_state
		self.drift = model_drift
		self.mean = model_mean
		self.covar = model_covar
		self.B = model_noise_matrix

		self.driving = driving_process
		self.mu_W = self.driving.get_mu_W()
		self.var_W = self.driving.get_var_W()

		self.H = observation_matrix
		self.obs_noise = observation_noise

		self.rng = rng


	def get_model_drift(self, interval):
		return self.drift(interval)


	def get_model_m(self, interval, jump_times, jump_sizes):
		return np.atleast_2d((self.mu_W * jump_sizes * self.mean(interval, jump_times)).sum(axis=-1)).T


	def get_model_S(self, interval, jump_times, jump_sizes):
		return (self.var_W * jump_sizes * self.covar(interval, jump_times)).sum(axis=-1)


	def get_model_B(self):
		return self.B


	def get_model_H(self):
		return self.H


	def get_model_kv(self):
		return self.obs_noise


	def get_model_var_W(self):
		return self.var_W


	def set_state(self, state):
		self.state = state


	def get_driving_jumps(self, rate, M=100, gamma_0=0.):
		return self.driving.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0)


	def increment_state(self, interval, M=100, gamma_0=0.):
		jump_times, jump_sizes = self.get_driving_jumps(rate=1./interval, M=M, gamma_0=gamma_0)
		m_vec = self.get_model_m(interval, jump_times, jump_sizes)
		S_mat = self.get_model_S(interval, jump_times, jump_sizes)
		try:
			C_mat = np.linalg.cholesky(S_mat)
			e = C_mat @ self.rng.normal(size=S_mat.shape[1])
		except np.linalg.LinAlgError:
			e = np.zeros(S_mat.shape[1])

		e = np.atleast_2d(e).T
		new_state = self.drift(interval) @ self.state + self.B @ e
		return new_state


	def observe_in_noise(self):
		return (self.H @ self.state + np.sqrt(self.var_W*self.obs_noise)*self.rng.normal()).item()

	def generate_observations(self, times):
		intervals = np.diff(times)
		observations = [self.observe_in_noise()]
		for diff in intervals:
			self.state = self.increment_state(diff)
			observations.append(self.observe_in_noise())
		return observations


class LangevinStateSpace(LinearSDEStateSpace):

	def __init__(self, initial_state, theta, driving_process, observation_matrix, observation_noise, rng=np.random.default_rng()):
		self.theta = theta
		super().__init__(initial_state, self.langevin_drift, self.langevin_mean, self.langevin_covar, self.langevin_noise_matrix, driving_process, observation_matrix, observation_noise, rng=rng)

	# model specific functors
	langevin_drift = lambda self, interval : np.exp(self.theta*interval)*np.array([[0.,1./self.theta],[0.,1.]]) + np.array([[1.,-1./self.theta],[0.,0.]])
	langevin_mean = lambda self, interval, jtime : np.exp(self.theta*(interval-jtime))*np.atleast_2d(np.array([[1./self.theta, 1]])).T + np.atleast_2d(np.array([[-1./self.theta, 0.]])).T
	langevin_covar = lambda self, interval, jtime : np.exp(2.*self.theta*(interval-jtime))*np.array([[1./(self.theta**2),1./self.theta],[1./self.theta,1.]])[:,:,np.newaxis] + np.exp(self.theta*(interval-jtime))*np.array([[-2./(self.theta**2),-1./self.theta],[-1./self.theta,0]])[:,:,np.newaxis] + np.array([[1./(self.theta**2),0.],[0.,0.]])[:,:,np.newaxis]
	langevin_noise_matrix = np.eye(2)




