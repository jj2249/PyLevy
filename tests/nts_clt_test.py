import numpy as np
import matplotlib.pyplot as plt
from PyLevy.utils.plotting_functions import qqplot, histogramplot
from PyLevy.processes import mean_mixture_processes
from PyLevy.utils.maths_functions import kstest, normDist, gammafnc
from tqdm import tqdm

t1 = 0.0
t2 = 1.0
kappa = 0.5
gamma = 1.35
delta = 1.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))
truncation = 1e-6
mu = 0.
mu_W = 1.
var_W = 2.

nSamples = 100000

endp = []
nts = mean_mixture_processes.NormalTemperedStableProcess(alpha=kappa, beta=beta, C=C , mu=mu, mu_W=mu_W, var_W=var_W)

for i in tqdm(range(nSamples)):
    nts_sample = nts.simulate_small_jumps(M=2000, rate=1./(t2-t1), truncation=truncation)
    endp.append(np.sum(nts_sample[1]))

endp = np.array(endp)
endp = (endp - np.mean(endp))/np.std(endp)

rvs = normDist.rvs(size=endp.shape[0])

pgf = True
titleqq = "Q-Q Plot for Residual Normal Tempered Stable Process with $\mu, \mu_{W}, \sigma_{W}, \kappa, \gamma, \delta, c =" + str(mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(kappa) + " ,"+ str(round(gamma,3)) + " ,"+ str(round(delta,3)) + " ,"+str(truncation) + "$"
qqplot(rvs, endp, xlabel="True Normal RVs", ylabel="Residual Normal Tempered Stable RVs", plottitle=titleqq, log=False, isPGF=pgf)
if pgf:
    plt.savefig("NormalTSCLTQQ.pgf", bbox_inches = "tight")
else:
    plt.show()

hist_axis = np.linspace(normDist.ppf(0.00001), normDist.ppf(0.99999), endp.shape[0])
pdf =normDist.pdf(hist_axis)

titlehist = "Histogram for Residual Normal Tempered Stable Process with $\mu, \mu_{W}, \sigma_{W}, \kappa, \gamma, \delta, c =" + str(mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(kappa) + " ,"+ str(round(gamma,3)) + " ,"+ str(round(delta,3)) + " ,"+str(truncation) + "$"
histogramplot(endp, pdf, hist_axis, num_bins = 200, xlabel="X", ylabel="PDF", plottitle=titlehist, isPGF=pgf)
if pgf:
    plt.savefig("NormalTSCLTHist.pgf", bbox_inches = "tight")
else:
    plt.show()