import numpy as np
import matplotlib.pyplot as plt
from PyLevy.utils.plotting_functions import plot_path
from PyLevy.processes import base_processes
from PyLevy.utils.maths_functions import gammafnc


kappa = 0.5
gamma = 1.35
delta = 1.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))
nPaths = 10

paths = []
time_ax = np.linspace(0., 1., 1000)
tsp = base_processes.TemperedStableProcess(alpha=kappa, beta=beta, C=C)

for _ in range(nPaths):
    tsp_sample = tsp.simulate_jumps(truncation=1e-10)
    tspintegral = tsp.integrate(time_ax, tsp_sample[0], tsp_sample[1])
    paths.append(tspintegral)

pgf =True
plot_path(time_ax, paths, title="10 Tempered Stable Paths with $\kappa, \gamma, \delta = " + str(kappa)+" ,"+ str(round(gamma, 3)) + " ," + str(delta) + "$")
if pgf:
    plt.savefig("TSPathSimulation.pgf", bbox_inches = "tight")
else:
    plt.show()

