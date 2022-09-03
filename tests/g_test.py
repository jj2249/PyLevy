import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes

plt.style.use('ggplot')

beta= 1.0
C = 10
nSamples = 1000

endp = []
g = base_processes.GammaProcess(beta=beta, C=C, truncation=1e-5)
samps = g.generate_marginal_samples(numSamples=nSamples)

fig, ax1 = plt.subplots(nrows=1, ncols=1)
axis = np.linspace(0., 1., nSamples)
for i in range(nSamples):
    g_sample = g.simulate_jumps(M=2000)
    endp.append(np.sum(g_sample[1]))
    print(i)
    #gigintegral = gig.integrate(axis, gig_sample[0], gig_sample[1])
    #endp.append(gigintegral[-1])

qqplot(endp, samps)
print(kstest(endp, samps))
plt.show()
