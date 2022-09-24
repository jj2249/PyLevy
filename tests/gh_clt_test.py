import numpy as np
import matplotlib.pyplot as plt
from PyLevy.utils.plotting_functions import qqplot, histogramplot
from PyLevy.processes import mean_mixture_processes
from PyLevy.utils.maths_functions import kstest, normDist
from tqdm import tqdm

t1 = 0.0
t2 = 1.0
delta = 1.3
gamma = np.sqrt(2.)
lambd = 0.2
mu = 0.
mu_W = 1.
var_W = 2.
truncation = 1e-6

nSamples = 40000

endp = []
gh = mean_mixture_processes.GeneralHyperbolicProcess(delta=delta, gamma=gamma, lambd=lambd, mu=mu, mu_W=mu_W, var_W=var_W)

fig, ax1 = plt.subplots(nrows=1, ncols=1)

for i in tqdm(range(nSamples)):
    gh_sample = gh.simulate_small_jumps(M=6000, rate=1. / (t2 - t1), truncation=truncation)
    endp.append(np.sum(gh_sample[1]))

endp = np.array(endp)
endp = (endp - np.mean(endp)) / np.std(endp)

rvs = normDist.rvs(size=endp.shape[0])

pgf = True
titleqq = "Q-Q Plot for Residual Generalised Hyperbolic Process with $\mu, \mu_{W}, \sigma_{W}, \delta, \gamma, \lambda, c =" + str(
    mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(delta) + " ," + str(round(gamma, 3)) + " ," + str(round(lambd, 3)) + " ," + str(truncation) + "$"
qqplot(rvs, endp, xlabel="True Normal RVs", ylabel="Residual Normal Tempered Stable RVs", plottitle=titleqq, log=False,
       isPGF=pgf)
if pgf:
    plt.savefig("GHCLTQQ.pgf", bbox_inches="tight")
else:
    plt.show()

hist_axis = np.linspace(normDist.ppf(0.00001), normDist.ppf(0.99999), endp.shape[0])
pdf = normDist.pdf(hist_axis)

titlehist = "Histogram for Residual Generalised Hyperbolic Process with $\mu, \mu_{W}, \sigma_{W}, \delta, \gamma, \lambda, c =" + str(
    mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(delta) + " ," + str(round(gamma, 3)) + " ," + str(round(lambd, 3)) + " ," + str(truncation) + "$"

histogramplot(endp, pdf, hist_axis, num_bins=200, xlabel="X", ylabel="PDF", plottitle=titlehist, isPGF=pgf)
if pgf:
    plt.savefig("GHCLTHist.pgf", bbox_inches="tight")
else:
    plt.show()
