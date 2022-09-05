import matplotlib.pyplot as plt
from PyLevy.processes import base_processes
from time import time

plt.style.use('ggplot')

alpha = .5
beta = 1.
C = 1.
sims = 100

# gp = base_processes.GammaProcess(beta=beta, C=C, truncation=1e-6)
nts = base_processes.TemperedStableProcess(alpha=alpha, beta=beta, C=C, truncation=1e-6)
t1 = time()
for _ in range(sims):
	_, jumps = nts.simulate_jumps(M=1000)
print(str(sims) + ': ' + str(time()-t1))
# fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

# ax1.hist(endp_gamma, bins=50, density=True)
# ax2.hist(endp_ts, bins=50, density=True)
# plt.show()
