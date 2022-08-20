from PyLevy.utils.maths_functions import logsumexp
from PyLevy.statespace.statespace import LinearSDEStateSpace


class FilterParticle:

	def __init__(self, prior_mean, prior_covar, transition_model, rng=np.random.default_rng()):
		self.__model = transition_model
		self.__logweight = 0.
		self.__a = prior_mean
		self.__C = prior_covar
		self.__B = self.__model.get_model_B()
		self.__H = self.__model.get_model_H()

	def __predict(self, interval):
		A = self.__model.get_model_drift(interval)
		m = self.__model.get_model_m(interval, jtimes, jsizes)
		S = self.__model.get_model_S(interval, jtimes, jsizes)


		jtimes, jsizes = self.__model.get_driving_jumps(interval)

		predicted_mean = A @ self.__a + m
		predicted_covar = A @ self.__C @ A.T + self.__B @ S @ self.__B.T

		return predicted_mean, predicted_covar


	def __correct(self, observation):
		obs_noise = self.__model.get_model_var_W() * self.__model.get_model_kv()
		K = np.atleast_2d((self.__C @ self.__H.T) / ((self.__H @ self.__C @ self.__H.T) + obs_noise))

		corrected_mean = self.__a + (K * (observation - self.__H @ self.__a))
		corrected_covar = self.__C - (K @ self.__H @ self.__C)

		return corrected_mean, corrected_covar


	# def __weight_update(self, observation):
	# 	obs_noise = self.__model.get_model_var_W() * self.__model.get_model_kv()
	# 	Cyt = self.__H @ self.__C @ self.__H.T + obs_noise
	# 	return -0.5*np.log(Cyt)


