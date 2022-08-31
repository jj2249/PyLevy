from PyLevy.utils.maths_functions import logsumexp
from PyLevy.statespace.statespace import LinearSDEStateSpace


class FilterParticle:

	def __init__(self, prior_mean, prior_covar, transition_model, rng=np.random.default_rng()):
		self.model = transition_model
		self.logweight = 0.
		self.a = prior_mean
		self.C = prior_covar
		self.B = self.model.get_model_B()
		self.H = self.model.get_model_H()

	def predict(self, interval):
		A = self.model.get_model_drift(interval)
		m = self.model.get_model_m(interval, jtimes, jsizes)
		S = self.model.get_model_S(interval, jtimes, jsizes)


		jtimes, jsizes = self.model.get_driving_jumps(interval)

		predicted_mean = A @ self.a + m
		predicted_covar = A @ self.C @ A.T + self.B @ S @ self.B.T

		return predicted_mean, predicted_covar


	def correct(self, observation):
		obs_noise = self.model.get_model_var_W() * self.model.get_model_kv()
		K = np.atleast_2d((self.C @ self.H.T) / ((self.H @ self.C @ self.H.T) + obs_noise))

		corrected_mean = self.a + (K * (observation - self.H @ self.a))
		corrected_covar = self.C - (K @ self.H @ self.C)

		return corrected_mean, corrected_covar


	# def __weight_update(self, observation):
	# 	obs_noise = self.__model.get_model_var_W() * self.__model.get_model_kv()
	# 	Cyt = self.__H @ self.__C @ self.__H.T + obs_noise
	# 	return -0.5*np.log(Cyt)


