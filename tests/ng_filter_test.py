from PyLevy.statespace.statespace import LinearSDEStateSpace, LangevinStateSpace
from PyLevy.processes.mean_mixture_processes import NormalGammaProcess
from PyLevy.filtering.filters import MarginalParticleFilter
from PyLevy.utils.plotting_functions import plot_filtering_results
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

theta = -1.
initial_state = np.atleast_2d(np.array([0., 0.])).T

observation_matrix = np.atleast_2d(np.array([1., 0.]))
observation_matrixd = np.atleast_2d(np.array([0., 1.]))

beta = 1.
C = 10.
mu = 0.
mu_W = 0.
var_W = 1.
noiseModel = 1
truncation = 1e-8

rngt = np.random.default_rng(seed=50)

rng = np.random.default_rng(seed=1)
ngp = NormalGammaProcess(beta, C, mu, mu_W, var_W, rng=rng)
langevin = LangevinStateSpace(initial_state, theta, ngp, observation_matrix, truncation_level=truncation,
                              modelCase=noiseModel, rng=rng)
times = rngt.exponential(scale=1., size=100).cumsum()
xs, ms, Ss = langevin.generate_observations(times, kv=1e-100)

rngd = np.random.default_rng(seed=1)
ngpd = NormalGammaProcess(beta, C, mu, mu_W, var_W, rng=rngd)
langevind = LangevinStateSpace(initial_state, theta, ngpd, observation_matrixd, truncation_level=truncation,
                               modelCase=noiseModel, rng=rngd)
xds, _, _ = langevind.generate_observations(times, kv=1e-100)

rng2 = np.random.default_rng(seed=100)
ngp = NormalGammaProcess(beta, C, mu, mu_W, var_W, rng=rng2)
langevin2 = LangevinStateSpace(initial_state, theta, ngp, observation_matrix, truncation_level=truncation,
                               modelCase=noiseModel, rng=rng2)

mpf = MarginalParticleFilter(np.zeros(2), np.eye(2), langevin2, rng=rng2, N=500)
means, covs = mpf.run_filter(times, xs, 1e-10, ms=ms, Ss=Ss, progbar=True)

plot_filtering_results(times, xs, xds, means)
