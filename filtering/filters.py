from PyLevy.utils.maths_functions import logsumexp
from PyLevy.statespace.statespace import LinearSDEStateSpace
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map
from functools import partial
from copy import copy


class KalmanFilter:

	def __init__(self, prior_mean, prior_covar, transition_model: LinearSDEStateSpace, rng=np.random.default_rng()):
		self.model = transition_model
		self.a = np.atleast_2d(prior_mean).T
		self.C = np.atleast_2d(prior_covar)
		self.B = self.model.get_model_B()
		self.H = self.model.get_model_H()


	def predict_given_jumps(self, interval, jtimes, jsizes):
		A = self.model.get_model_drift(interval)
		m = self.model.get_model_m(interval, jtimes, jsizes)
		S = self.model.get_model_S(interval, jtimes, jsizes)

		predicted_mean = A @ self.a + m
		predicted_covar = A @ self.C @ A.T + self.B @ S @ self.B.T

		return predicted_mean, predicted_covar


	def correct(self, observation, kv):
		obs_noise = self.model.get_model_var_W() * kv
		K = np.atleast_2d((self.C @ self.H.T) / ((self.H @ self.C @ self.H.T) + obs_noise))

		corrected_mean = self.a + (K * (observation - self.H @ self.a))
		corrected_covar = self.C - (K @ self.H @ self.C)

		return corrected_mean, corrected_covar


class FilterParticle(KalmanFilter):

	def __init__(self, prior_mean, prior_covar, transition_model: LinearSDEStateSpace, rng=np.random.default_rng()):
		super().__init__(prior_mean, prior_covar, transition_model, rng)


	def predict(self, interval):
		jtimes, jsizes = self.model.get_driving_jumps(interval)
		return self.predict_given_jumps(interval, jtimes, jsizes)


	def lweight_update(self, observation, kv):
		obs_noise = self.model.get_model_var_W() * kv
		Cyt = self.H @ self.C @ self.H.T + obs_noise
		return -0.5*np.log(Cyt).item()


	def increment(self, interval, observation, kv):
		self.a, self.C = self.predict(interval)
		self.a, self.C = self.correct(observation, kv)
		return self.lweight_update(observation, kv)


class MarginalParticleFilter:

	def __init__(self, prior_mean, prior_covar, transition_model: LinearSDEStateSpace, N=100, resample_rate=0.5, rng=np.random.default_rng()):
		self.log_resample_limit = np.log(N*resample_rate)
		self.kalmans = np.array([FilterParticle(prior_mean, prior_covar, transition_model, rng=rng) for _ in range(N)])
		self.lweights = np.atleast_2d(np.zeros(N))
		self.N = N
		self.P = prior_mean.shape[0]


	def normalise_weights(self):
		lsum_weights = logsumexp(self.lweights, lambda x: 1., np.ones(self.N), retlog=True)
		lweights = self.lweights - lsum_weights
		return lweights


	def get_logDninf(self):
		return -np.max(self.lweights)


	@staticmethod
	def particle_increment(particle, interval, observation, kv):
		return particle.increment(interval, observation, kv)


	def increment_all_particles(self, interval, observation, kv):
		lweight_updates = t_map(partial(self.particle_increment, interval=interval, observation=observation, kv=kv), self.kalmans, disable=True)
		return self.lweights + np.atleast_2d(lweight_updates)


	def get_state_posterior(self):
		eX = np.array([particle.a for particle in self.kalmans])
		msum = logsumexp(self.lweights, lambda x: x, eX, axis=0, retlog=False)
		eXXt = np.array([particle.C + (particle.a @ particle.a.T) for particle in self.kalmans])
		Covsum = logsumexp(self.lweights, lambda x: x, eXXt, axis=0, retlog=False) - msum @ msum.T
		return msum, Covsum


	def resample_particles(self):
		probabilites = np.exp(self.lweights)
		probabilites = np.nan_to_num(probabilites)
		probabilites = probabilites / np.sum(probabilites)

		selections = self.rng.multinomial(self.N, probabilites)

		new_particles = []

		for idx in range(self.N):
			for _ in range(selections[idx]):
				new_particles.append(copy.copy(self.particles[idx]))
		return -np.log(self.N)*np.atleast_2d(np.ones(self.N)), new_particles



	def run_filter(self, times, observations, kv, progbar=False):
		curr_t = 0.
		means = np.empty((self.P, 1, times.shape[0]))
		covs = np.empty((self.P, self.P, times.shape[0]))
		for idx, (time, observation) in tqdm(enumerate(list(zip(times, observations))), disable=not progbar):
			self.lweights = self.increment_all_particles(time-curr_t, observation, kv)
			self.lweights = self.normalise_weights()
			curr_t = time
			
			if self.get_logDninf() < self.log_resample_limit:
				self.lweights, self.kalmans = self.resample_particles()

			mean, cov = self.get_state_posterior()
			means[:, :, idx] = mean
			covs[:, :, idx] = cov

		return np.squeeze(means), covs
