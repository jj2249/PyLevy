import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes

plt.style.use('ggplot')

t1 = 1e-1
t2 = 2.01e-1
beta = 1.0
C = 10.
nSamples = 1000

endp = []
g = base_processes.GammaProcess(beta=beta, C=C)
samps = g.generate_marginal_samples(numSamples=nSamples, tHorizon=t2 - t1)

fig, ax1 = plt.subplots(nrows=1, ncols=1)

for i in range(nSamples):
    g_sample = g.simulate_jumps(M=2000, rate=1./(t2-t1), truncation=1e-8)
    endp.append(np.sum(g_sample[1]))

qqplot(endp, samps)
print(kstest(endp, samps))
plt.show()
