from PyLevy.statespace.statespace import LinearSDEStateSpace, LangevinStateSpace
from PyLevy.processes.mean_mixture_processes import NormalTemperedStableProcess
from PyLevy.filtering.filters import MarginalParticleFilter
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

theta = -.2
initial_state = np.atleast_2d(np.array([0., 0.])).T

observation_matrix = np.atleast_2d(np.array([1., 0.]))
observation_matrixd = np.atleast_2d(np.array([0., 1.]))

alpha = 0.8
beta = 1.
C = 1.
truncation = 1e-6
mu = 0.
mu_W = 0.
var_W = 1.
noiseModel = 2

rng = np.random.default_rng(seed=2)
ngp = NormalTemperedStableProcess(alpha= alpha, beta=beta, C=C, mu=mu, mu_W=mu_W, var_W=var_W, rng=rng)
langevin = LangevinStateSpace(initial_state, theta, ngp, observation_matrix, modelCase=noiseModel, rng=rng)
times = rng.exponential(size=100).cumsum()
xs = langevin.generate_observations(times, kv=1e-1)

rngd = np.random.default_rng(seed=2)
ngpd = NormalTemperedStableProcess(alpha= alpha, beta=beta, C=C, mu=mu, mu_W=mu_W, var_W=var_W, rng=rngd)
langevind = LangevinStateSpace(initial_state, theta, ngpd, observation_matrixd, modelCase=noiseModel, rng=rngd)
xds = langevind.generate_observations(times, kv=1e-5)


rng2 = np.random.default_rng(seed=100)

ngp2 = NormalTemperedStableProcess(alpha= alpha, beta=beta, C=C, mu=mu, mu_W=mu_W, var_W=var_W, rng=rng2)
langevin2 = LangevinStateSpace(initial_state, theta, ngp2, observation_matrix, modelCase=noiseModel, rng=rng2, truncation_level=1e-3)
mpf = MarginalParticleFilter(np.zeros(2), np.eye(2), langevin2, rng=rng2, N=500)
means, covs = mpf.run_filter(times, xs, 1e-1, progbar=True)

fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(times, xs)
ax2.plot(times, xds)
ax1.plot(times, means[0])
ax2.plot(times, means[1])
plt.show()