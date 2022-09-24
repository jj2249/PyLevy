import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes
from tqdm import tqdm

delta = 1.3
gamma = np.sqrt(2.)
lambd = .2
nSamples = 10000


endp = []
gig = base_processes.GIGProcess(delta=delta, gamma=gamma, lambd=lambd)
samps = gig.generate_marginal_samples(numSamples=nSamples)

fig, ax1 = plt.subplots(nrows=1, ncols=1)
axis = np.linspace(0., 1., nSamples)
for i in tqdm(range(nSamples)):
    gig_sample = gig.simulate_jumps(M=2000, truncation=1e-6)
    endpoint = np.sum(gig_sample[1])
    endp.append(endpoint)

pgf = True
title = "Q-Q plot for GIG Process with $\delta, \gamma, \lambda = " + str(delta) + " ," + str(round(gamma, 3)) + " ," + str(lambd) + "$"
qqplot(samps, endp, xlabel="True RVs", ylabel="GIG Random Variables at $t = T_{horizon}$", log=True, plottitle=title, isPGF=pgf)
if pgf:
    plt.savefig("GIGSimulationQQPlot.pgf", bbox_inches = "tight")
else:
    plt.show()