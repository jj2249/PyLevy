import numpy as np
import matplotlib.pyplot as plt
from PyLevy.processes import base_processes

plt.style.use('ggplot')

delta = 3.
gamma = 0.5
lambd = -1.

endp = []
gig = base_processes.GIGProcess(delta=delta, gamma=gamma, lambd=lambd)
# samps = gig.marginal_samples(1000, 1.)

fig, ax1 = plt.subplots(nrows=1, ncols=1)
axis = np.linspace(0., 1., 1000)
for i in range(1000):
	gig_sample = gig.simulate_jumps()
	gigintegral = gig.integrate(axis, gig_sample[0], gig_sample[1])
	endp.append(gigintegral[-1])
ax1.hist(endp, density=True, bins=50)
# ax2.hist(samps, density=True, bins=50)
plt.show()