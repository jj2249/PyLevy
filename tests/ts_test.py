import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from tqdm import tqdm
from PyLevy.utils.maths_functions import gammafnc, levy_stable
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes

plt.style.use('ggplot')

t1 = 0.0
t2 = 1.0
kappa = 0.5
gamma = 1.35
delta = 1.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))

ts = base_processes.TemperedStableProcess(alpha=kappa, beta=beta, C=C)
nSamples = 10000
endp = []

fig, ax1 = plt.subplots(nrows=1, ncols=1)
for i in tqdm(range(nSamples)):
    _, ts_sample = ts.simulate_jumps(M=2000,  truncation=1e-10)
    endp.append(np.sum(ts_sample))
samps = ts.generate_marginal_samples(numSamples=nSamples, tHorizon=t2-t1)

pgf =True
title = "Q-Q plot for TS Process with $\kappa, \gamma, \delta = " + str(kappa)+" ,"+ str(round(gamma, 3)) + " ," + str(delta) + "$"
qqplot(samps, endp, xlabel="True RVs", ylabel="TS Random Variables at $t = T_{horizon}$", plottitle=title, isPGF=pgf)
if pgf:
    plt.savefig("TSSimulationQQPlot.pgf", bbox_inches = "tight")
else:
    plt.show()