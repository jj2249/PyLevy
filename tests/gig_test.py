import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes

plt.style.use('ggplot')

delta = 3.
gamma = 0.5
lambd = 0.49
nSamples = 100

endp = []
gig = base_processes.GIGProcess(delta=delta, gamma=gamma, lambd=lambd)
samps = gig.generate_marginal_samples(numSamples=nSamples)
# gig.simulate_jumps()
fig, ax1 = plt.subplots(nrows=1, ncols=1)
axis = np.linspace(0., 1., nSamples)
for i in range(nSamples):
    gig_sample = gig.simulate_jumps(M=10000, truncation=1e-3)
    endpoint = np.sum(gig_sample[1])
    endp.append(endpoint)

qqplot(endp, samps)
print(kstest(endp, samps))
plt.show()