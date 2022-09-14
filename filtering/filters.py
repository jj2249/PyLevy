from PyLevy.utils.maths_functions import logsumexp, log, gammafnc, exp
from PyLevy.statespace.statespace import LinearSDEStateSpace
from scipy.stats import kstest
from PyLevy.utils.plotting_functions import qqplot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map
from functools import partial
from copy import deepcopy

import numpy as np


class KalmanFilter:

    def __init__(self, prior_mean, prior_covar, transition_model: LinearSDEStateSpace):
        self.model = transition_model
        self.a = np.atleast_2d(prior_mean).T
        self.C = prior_covar
        self.B = self.model.get_model_B()
        self.H = self.model.get_model_H()

    def predict_given_jumps(self, interval, jtimes, jsizes, m=None, S=None):
        A = self.model.get_model_drift(interval)
        if m is None:
            m = self.model.get_model_m(interval=interval, jump_times=jtimes, jump_sizes=jsizes)
            S = self.model.get_model_S(interval=interval, jump_times=jtimes, jump_sizes=jsizes)

        full_noise_covar = S + self.model.get_model_Ce(interval)
        predicted_mean = A @ self.a + m.reshape((2, 1))
        predicted_covar = A @ self.C @ A.T + self.B @ full_noise_covar @ self.B.T

        return predicted_mean, predicted_covar

    def correct(self, observation, kv):
        obs_noise = self.model.get_model_var_W() * kv
        K = np.atleast_2d((self.C @ self.H.T) / ((self.H @ self.C @ self.H.T) + obs_noise))
        corrected_mean = self.a + (K * (observation - (self.H @ self.a).squeeze()))
        corrected_covar = self.C - (K @ self.H @ self.C)

        return corrected_mean, corrected_covar


class FilterParticle(KalmanFilter):

    def __init__(self, prior_mean, prior_covar, transition_model: LinearSDEStateSpace):
        super().__init__(prior_mean, prior_covar, transition_model)

    def predict(self, interval, m=None, S=None):
        jtimes, jsizes = self.model.get_driving_jumps(rate=1. / interval)
        return self.predict_given_jumps(interval, m=m, S=S, jtimes=jtimes, jsizes=jsizes)

    def lweight_update(self, observation, kv):
        obs_noise = self.model.get_model_var_W() * kv
        F_N = np.squeeze(self.H @ self.C @ self.H.T + obs_noise)
        wt = observation - np.squeeze(self.H @ self.a)
        return -0.5 * log(abs(F_N)) - 0.5 * (wt ** 2) / (self.model.get_model_var_W() * F_N)

    def increment(self, interval, observation, kv, m=None, S=None):
        self.a, self.C = self.predict(interval, m=m, S=S)
        self.a, self.C = self.correct(observation, kv)
        return self.lweight_update(observation, kv)


class MarginalParticleFilter:

    def __init__(self, prior_mean, prior_covar, transition_model: LinearSDEStateSpace, N=500, resample_rate=0.5,
                 rng=np.random.default_rng()):
        self.log_resample_limit = np.log(N * resample_rate)
        self.kalmans = np.array([FilterParticle(prior_mean, prior_covar, transition_model=transition_model) for _ in range(N)])
        self.lweights = np.atleast_2d(np.zeros(N))
        self.N = N
        self.P = prior_mean.shape[0]
        self.rng = rng

    def normalise_weights(self):
        lsum_weights = logsumexp(self.lweights, lambda x: 1., np.ones(self.N), retlog=True)
        return self.lweights - lsum_weights

    def get_logDninf(self):
        return -np.max(self.lweights)

    @staticmethod
    def particle_increment(particle, interval, observation, kv, m=None, S=None):
        return particle.increment(interval=interval, m=m, S=S, observation=observation, kv=kv)

    def increment_all_particles(self, interval, observation, kv, m=None, S=None):
        lweight_updates = t_map(
            partial(self.particle_increment, m=m, S=S, interval=interval, observation=observation, kv=kv), self.kalmans,
            disable=True)
        return self.lweights + np.atleast_2d(lweight_updates)

    def get_state_posterior(self):
        eX = np.array([particle.a for particle in self.kalmans])
        msum = logsumexp(self.lweights, lambda x: x, eX, axis=0, retlog=False)
        eXXt = np.array(
            [particle.C + (particle.a @ particle.a.T) for particle in self.kalmans])
        Covsum = logsumexp(self.lweights, lambda x: x, eXXt, axis=0, retlog=False) - msum @ msum.T
        return msum, Covsum

    def resample_particles(self):
        """ Adapted from filterpy.monte_carlo.stratified_resample """
        N_p = self.N
        u = np.zeros((N_p, 1))
        c = np.cumsum(exp(self.lweights))
        c[-1] = 1.0
        i = 0
        u[0] = self.rng.random() / N_p
        new_kfs = [0] * N_p
        for j in range(N_p):

            u[j] = u[0] + j / N_p

            while u[j] > c[i]:
                i = i + 1

            new_kfs[j] = deepcopy(self.kalmans[i])

        log_weights = np.array([-np.log(N_p)] * N_p)
        return np.atleast_2d(log_weights), new_kfs

    def run_filter(self, times, observations, kv, ms, Ss, progbar=False):
        curr_t = times[0]
        means = np.zeros((self.P, 1, times.shape[0]))
        covs = np.empty((self.P, self.P, times.shape[0]))
        normRV = np.atleast_2d(self.rng.normal(size=self.P)).T
        means[:, :, 0] = np.zeros(shape=(self.P,1))
        covs[:, :, 0] = np.eye(self.P)
        for idx in tqdm(range(1, times.shape[0]), disable=not progbar):
            if self.get_logDninf() < self.log_resample_limit:
                self.lweights, self.kalmans = self.resample_particles()
            self.lweights = self.increment_all_particles(times[idx] - curr_t, observation=observations[idx], kv=kv)
            self.lweights = self.normalise_weights()
            curr_t = times[idx]
            mean, cov = self.get_state_posterior()
            means[:, :, idx] = mean
            covs[:, :, idx] = cov

        return np.squeeze(means), covs
