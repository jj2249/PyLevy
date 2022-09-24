import numpy as np
import matplotlib.pyplot as plt
from PyLevy.utils.plotting_functions import plot_path
from PyLevy.processes import base_processes

delta = 1.3
gamma = np.sqrt(2.)
lambd = 0.2
nPaths = 10

paths = []
time_ax = np.linspace(0., 1., 1000)
gigp = base_processes.GIGProcess(delta=delta, gamma=gamma, lambd=lambd)

for _ in range(nPaths):
    gigp_sample = gigp.simulate_jumps(truncation=1e-6)
    gigpintegral = gigp.integrate(time_ax, gigp_sample[0], gigp_sample[1])
    paths.append(gigpintegral)

pgf = True
plot_path(time_ax, paths,
          title="10 GIG Paths with $\delta, \gamma, \lambda = " + str(delta) + " ," + str(round(gamma, 3)) + " ," + str(
              lambd) + "$")
if pgf:
    plt.savefig("GIGPathSimulation.pgf", bbox_inches="tight")
else:
    plt.show()
