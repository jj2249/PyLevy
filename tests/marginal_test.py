import matplotlib.pyplot as plt
from PyLevy.processes import base_processes

plt.style.use('ggplot')

alpha = .5
beta = 1.
C = 1.
sims = 1000

gp = base_processes.GammaProcess(beta=beta, C=C, truncation=1e-6)
tsp = base_processes.TemperedStableProcess(alpha=alpha, beta=beta, C=C, truncation=1e-6)

endp_gamma = []
endp_ts = []

for _ in range(sims):
	_, jumps_gamma = gp.simulate_jumps()
	endp_gamma.append(jumps_gamma.sum())

	_, jumps_ts = tsp.simulate_jumps()
	endp_ts.append(jumps_ts.sum())

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

ax1.hist(endp_gamma, bins=50, density=True)
ax2.hist(endp_ts, bins=50, density=True)
plt.show()
