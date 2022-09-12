from PyLevy.statespace.statespace import LinearSDEStateSpace, LangevinStateSpace
from PyLevy.processes.mean_mixture_processes import NormalTemperedStableProcess
from PyLevy.filtering.filters import MarginalParticleFilter
from PyLevy.utils.plotting_functions import plot_filtering_results
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

theta = -.6
initial_state = np.atleast_2d(np.array([0., 0.])).T

observation_matrix = np.atleast_2d(np.array([1., 0.]))
observation_matrixd = np.atleast_2d(np.array([0., 1.]))

alpha = 0.1
beta = 10.
C = .1
mu = 0.
mu_W = 0.0
var_W = 1.
noiseModel = 1
truncation = 1e-8

rngt = np.random.default_rng(seed=50)

rng = np.random.default_rng(seed=222)
ngp = NormalTemperedStableProcess(alpha, beta, C, mu, mu_W, var_W, rng=rng)
langevin = LangevinStateSpace(initial_state, theta, ngp, observation_matrix, truncation_level=truncation,
                              modelCase=noiseModel, rng=rng)

times = rngt.exponential(size=100).cumsum()
xs, _, _ = langevin.generate_observations(times, kv=1e-10)

rngd = np.random.default_rng(seed=222)
ngpd = NormalTemperedStableProcess(alpha, beta, C, mu, mu_W, var_W, rng=rngd)
langevind = LangevinStateSpace(initial_state, theta, ngpd, observation_matrixd, truncation_level=truncation,
                               modelCase=noiseModel, rng=rngd)
xds, m, S = langevind.generate_observations(times, kv=1e-15)

rng2 = np.random.default_rng(seed=120)
ngp = NormalTemperedStableProcess(alpha, beta, C, mu, mu_W, var_W, rng=rng2)
langevin2 = LangevinStateSpace(initial_state, theta, ngp, observation_matrix, truncation_level=truncation,
                               modelCase=noiseModel, rng=rng2)

mpf = MarginalParticleFilter(np.zeros(2), var_W*np.eye(2), langevin2, rng=rng2, N=500)
means, covs = mpf.run_filter(times, xs, 1e-10, ms=m, Ss=S, progbar=True)

plot_filtering_results(times, xs, xds, means)
