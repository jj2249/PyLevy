import numpy as np
import matplotlib.pyplot as plt
from PyLevy.processes import base_processes

plt.style.use('ggplot')

alpha = .5
beta = 1.
C = 1.

gp = base_processes.GammaProcess(beta=beta, C=C)
tsp = base_processes.TemperedStableProcess(alpha=alpha, beta=beta, C=C)

gammap_sample = gp.simulate_jumps(truncation=1e-6)
tsp_sample = tsp.simulate_jumps(truncation=1e-6)

axis = np.linspace(0., 1., 1000)

gpintegral = gp.integrate(axis, gammap_sample[0], gammap_sample[1])
tspintegral = tsp.integrate(axis, tsp_sample[0], tsp_sample[1])

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

ax1.step(axis, gpintegral, lw=1.2)
ax2.step(axis, tspintegral, lw=1.2)
plt.show()