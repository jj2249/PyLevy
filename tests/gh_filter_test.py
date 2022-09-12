from PyLevy.statespace.statespace import LinearSDEStateSpace, LangevinStateSpace
from PyLevy.processes.mean_mixture_processes import GeneralHyperbolicProcess
from PyLevy.filtering.filters import MarginalParticleFilter
from PyLevy.utils.plotting_functions import plot_filtering_results
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

theta = -.9
initial_state = np.atleast_2d(np.array([0., 0.])).T

observation_matrix = np.atleast_2d(np.array([1., 0.]))
observation_matrixd = np.atleast_2d(np.array([0., 1.]))

delta = 1.
gamma = 0.5
lambd = -1.
mu = 0.
mu_W = 0.
var_W = 1.
noiseModel = 1
truncation = 1e-6

rngt = np.random.default_rng(seed=50)

rng = np.random.default_rng(seed=4)
ghp = GeneralHyperbolicProcess(delta=delta, lambd=lambd, gamma=gamma, mu=mu, mu_W=mu_W, var_W=var_W, rng=rng)
langevin = LangevinStateSpace(initial_state, theta, ghp, observation_matrix, truncation_level=truncation,
                              modelCase=noiseModel, rng=rng)
times = rngt.exponential(size=100).cumsum()
xs,_,_ = langevin.generate_observations(times, kv=1e-15)

rngd = np.random.default_rng(seed=4)
ghpd = GeneralHyperbolicProcess(delta=delta, lambd=lambd, gamma=gamma, mu=mu, mu_W=mu_W, var_W=var_W, rng=rng)
langevind = LangevinStateSpace(initial_state, theta, ghpd, observation_matrixd, truncation_level=truncation,
                               modelCase=noiseModel, rng=rngd)
xds, ms, Ss = langevind.generate_observations(times, kv=1e-15)

rng2 = np.random.default_rng(seed=2)
ghp = GeneralHyperbolicProcess(delta=delta, lambd=lambd, gamma=gamma, mu=mu, mu_W=mu_W, var_W=var_W, rng=rng)
langevin2 = LangevinStateSpace(initial_state, theta, ghp, observation_matrix, truncation_level=truncation,
                               modelCase=noiseModel, rng=rng2)

mpf = MarginalParticleFilter(np.zeros(2), np.eye(2), langevin2, rng=rng2, N=500)
means, covs = mpf.run_filter(times, xs, 1e-15, ms=ms, Ss=Ss, progbar=True)

plot_filtering_results(times, xs, xds, means)

