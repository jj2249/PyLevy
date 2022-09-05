import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes

plt.style.use('ggplot')

delta = 3.
gamma = 0.5
lambd = -0.2
nSamples = 1000

endp = []
gig = base_processes.GIGProcess(delta=delta, gamma=gamma, lambd=lambd)
samps = gig.generate_marginal_samples(numSamples=nSamples)
# gig.simulate_jumps()
fig, ax1 = plt.subplots(nrows=1, ncols=1)
axis = np.linspace(0., 1., nSamples)
for i in range(nSamples):
    gig_sample = gig.simulate_jumps()
    endpoint = np.sum(gig_sample[1])
    endp.append(endpoint)

qqplot(endp, samps)
print(kstest(endp, samps))
plt.show()

"""
def blah():
    min_jump = np.inf
    x = []
    curr_epoch = gamma_0
    while min_jump >= truncation:
        epoch_seq = self.rng.exponential(scale=rate, size=M)
        epoch_seq[0] += curr_epoch
        epoch_seq = epoch_seq.cumsum()
        curr_epoch = epoch_seq[-1]
        x_seq = h_func(epoch_seq)
        min_jump = x_seq[-1]
        if min_jump < truncation:
            x.append(x_seq[x_seq >= truncation])
        else:
            x.append(x_seq)
    x = np.concatenate(x)
    acceptance_seq = thinning_func(x)
    u = self.rng.uniform(low=0., high=1., size=x.size)
    x = x[u < acceptance_seq]
    jtimes = self.rng.uniform(low=0., high=1. / rate, size=x.size)
    return jtimes, x

"""