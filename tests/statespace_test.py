from PyLevy.statespace.statespace import LinearSDEStateSpace, LangevinStateSpace
from PyLevy.processes.mean_mixture_processes import NormalGammaProcess, NormalTemperedStableProcess
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

theta = -0.2
initial_state = np.atleast_2d(np.array([0., 0.])).T

observation_matrix1 = np.atleast_2d(np.array([1., 0.]))
observation_matrix2 = np.atleast_2d(np.array([0., 1.]))

alpha = 1.5
beta = 1.
C = 1.
mu = 0.
mu_W = 0.
var_W = 1.

rng1 = np.random.default_rng(1)
rng2 = np.random.default_rng(1)

ngp1 = NormalGammaProcess(beta, C, mu, mu_W, var_W, rng=rng1)
# ngp1 = NormalTemperedStableProcess(alpha, beta, C, mu, mu_W, var_W, rng=rng1)
langevin1 = LangevinStateSpace(initial_state, theta, ngp1, observation_matrix1, 1e-1, rng=rng1)
ngp2 = NormalGammaProcess(beta, C, mu, mu_W, var_W, rng=rng2)
# ngp2 = NormalTemperedStableProcess(alpha, beta, C, mu, mu_W, var_W, rng=rng2)
langevin2 = LangevinStateSpace(initial_state, theta, ngp2, observation_matrix2, 1e-15, rng=rng2)
# times = np.random.rand(500).cumsum()
times = np.random.exponential(size=500).cumsum()
xs = langevin1.generate_observations(times)
xdots = langevin2.generate_observations(times)

fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(times, xs)
ax2.plot(times, xdots)
plt.show()