from sys import argv
import matplotlib.pyplot as plt
from PyLevy.processes import base_processes


plt.style.use('ggplot')

alpha = float(argv[1])
beta = float(argv[2])
C = float(argv[3])
sims = int(argv[4])

gp = processes.GammaProcess(beta=beta, C=C)
tsp = processes.TemperedStableProcess(alpha=alpha, beta=beta, C=C)

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
