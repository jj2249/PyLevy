from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from PyLevy.processes import processes

plt.style.use('ggplot')

alpha = float(argv[1])
beta = float(argv[2])
C = float(argv[3])
sims = int(argv[4])

gp = processes.GammaProcess(beta=beta, C=C)
tsp = processes.TemperedStableProcess(alpha=alpha, beta=beta, C=C)

gammap_sample = gp.simulate_jumps()
tsp_sample = tsp.simulate_jumps()

axis = np.linspace(0., 1., 1000)

gpintegral = gp.integrate(axis, gammap_sample[0], gammap_sample[1])
tspintegral = tsp.integrate(axis, tsp_sample[0], tsp_sample[1])

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

ax1.step(axis, gpintegral, lw=1.2)
ax2.step(axis, tspintegral, lw=1.2)
plt.show()