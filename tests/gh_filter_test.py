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

rng = np.random.default_rng(seed=1)
ghp = GeneralHyperbolicProcess(delta=delta, lambd=lambd, gamma=gamma, mu=mu, mu_W=mu_W, var_W=var_W, rng=rng)
langevin = LangevinStateSpace(initial_state, theta, ghp, observation_matrix, truncation_level=truncation,
                              modelCase=noiseModel, rng=rng)
times = rngt.exponential(size=100).cumsum()
x, ms, Ss = langevin.generate_observations(times, kv=1e-1)


observations = x[0]
xs = x[1]
xds = x[2]

rng2 = np.random.default_rng(seed=100)
ghp = GeneralHyperbolicProcess(delta=delta, lambd=lambd, gamma=gamma, mu=mu, mu_W=mu_W, var_W=var_W, rng=rng)
langevin2 = LangevinStateSpace(initial_state, theta, ghp, observation_matrix, truncation_level=truncation,
                               modelCase=noiseModel, rng=rng2)

kv = 1e-1
mpf = MarginalParticleFilter(np.zeros(2), np.eye(2), transition_model=langevin2, rng=rng2, N=500)
means, covs = mpf.run_filter(times, observations, kv, ms=ms, Ss=Ss, progbar=True)

stds = [covs[0,0], covs[1,1]]

plot_filtering_results(times, observations, xs, xds, means, stds)
