import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from PyLevy.utils.maths_functions import gammafnc, levy_stable
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes


def marginal_samples(numSamples, tHorizon=1.0):
    x = levy_stable.rvs(kappa, beta=1.0, loc=0.0, scale=(tHorizon * delta) ** (1 / kappa),
                        size=numSamples)
    beta = 0.5 * gamma ** (1 / kappa)
    prob_acc = np.exp(-beta * x)
    us = np.random.uniform(0., 1., size=prob_acc.size)
    xs = np.where(prob_acc > us, x, 0.)
    return xs[xs > 0.]

plt.style.use('ggplot')

kappa = 0.5
gamma = 2.0
delta = 1.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))
ts = base_processes.TemperedStableProcess(alpha=kappa, beta=beta, C=C)

nSamples = 10000
endp = []
fig, ax1 = plt.subplots(nrows=1, ncols=1)
axis = np.linspace(0., 1., nSamples)
for i in range(nSamples):
    _, ts_sample = ts.simulate_jumps(M=2000)
    endp.append(np.sum(ts_sample))
# gigintegral = gig.integrate(axis, gig_sample[0], gig_sample[1])
# endp.append(gigintegral[-1])

samps = marginal_samples(numSamples=nSamples)
qqplot(endp, samps)
print(kstest(endp, samps))
plt.show()
