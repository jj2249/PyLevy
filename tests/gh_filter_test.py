from PyLevy.statespace.statespace import LinearSDEStateSpace, LangevinStateSpace
from PyLevy.processes.mean_mixture_processes import GeneralHyperbolicProcess
from PyLevy.filtering.filters import MarginalParticleFilter
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

theta = -.2
initial_state = np.atleast_2d(np.array([0., 0.])).T

observation_matrix = np.atleast_2d(np.array([1., 0.]))
observation_matrixd = np.atleast_2d(np.array([0., 1.]))

gamma = 1.5
delta = 1.
lambd = -1.
mu = 0.
mu_W = 0.
var_W = 1.
noiseModel = 3
truncation = 0.

rng = np.random.default_rng(seed=2)
ngp = GeneralHyperbolicProcess(delta=delta, lambd= lambd, gamma=gamma, mu= mu, mu_W= mu_W, var_W= var_W, truncation=truncation, rng=rng)
langevin = LangevinStateSpace(initial_state, theta, ngp, observation_matrix, modelCase=noiseModel, rng=rng)
times = rng.exponential(size=100).cumsum()
xs = langevin.generate_observations(times, kv=1e-5)

rngd = np.random.default_rng(seed=2)
ngpd = GeneralHyperbolicProcess(delta=delta, lambd= lambd, gamma=gamma, mu= mu, mu_W= mu_W, var_W= var_W, truncation=truncation, rng=rng)
langevind = LangevinStateSpace(initial_state, theta, ngpd, observation_matrixd, modelCase=noiseModel, rng=rngd)
xds = langevind.generate_observations(times, kv=1e-5)


rng2 = np.random.default_rng(seed=2)
ngp = GeneralHyperbolicProcess(delta=delta, lambd= lambd, gamma=gamma, mu= mu, mu_W= mu_W, var_W= var_W, truncation=truncation, rng=rng)
langevin2 = LangevinStateSpace(initial_state, theta, ngp, observation_matrix, modelCase=noiseModel, rng=rng2)

mpf = MarginalParticleFilter(np.zeros(2), np.eye(2), langevin2, rng=rng2, N=100)
means, covs = mpf.run_filter(times, xs, 1e-5, progbar=True)

fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(times, xs)
ax2.plot(times, xds)
ax1.plot(times, means[0])
ax2.plot(times, means[1])
plt.show()