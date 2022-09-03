import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from PyLevy.utils.maths_functions import gammafnc, levy_stable
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes

plt.style.use('ggplot')

t1 = 0.0
t2= 1.0
kappa = 0.1
gamma = 1.
delta = 2.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))
print(kappa, beta, C)
ts = base_processes.TemperedStableProcess(alpha=kappa, beta=beta, C=C, truncation=0.)
nSamples = 10000
endp = []
fig, ax1 = plt.subplots(nrows=1, ncols=1)
axis = np.linspace(t1, t2, nSamples)
for i in range(nSamples):
    _, ts_sample = ts.simulate_jumps(M=2000)
    endp.append(np.sum(ts_sample))
    print(i)
    # gigintegral = gig.integrate(axis, gig_sample[0], gig_sample[1])
    # endp.append(gigintegral[-1])

samps = ts.generate_marginal_samples(numSamples=nSamples, tHorizon=t2-t1)
qqplot(samps, endp)
print(kstest(samps, endp))
plt.show()
