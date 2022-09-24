import numpy as np
import matplotlib.pyplot as plt
from PyLevy.utils.plotting_functions import histogramplot
from PyLevy.processes import base_processes
from PyLevy.utils.maths_functions import kstest, gammaDist
from tqdm import tqdm


t1 = 0.0
t2 = 1.0
gamma = np.sqrt(2.)
beta = gamma**2/2.
nu = 2.
nSamples = 100000

endp = []
g = base_processes.GammaProcess(beta=beta, C=nu)

fig, ax1 = plt.subplots(nrows=1, ncols=1)

for i in tqdm(range(nSamples)):
    g_sample = g.simulate_jumps(M=2000, rate=1./(t2-t1), truncation=1e-10)
    endp.append(np.sum(g_sample[1]))

pdf = gammaDist.pdf(x=np.linspace(min(endp), max(endp), len(endp)), a=nu, loc=0., scale=1/beta)

pgf =True
histogramplot(endp, pdf, xlabel="X", ylabel="PDF", plottitle="Histogram for Gamma Process", isPGF=pgf)
if pgf:
    plt.savefig("GammaPathHistogram.pgf", bbox_inches = "tight")
else:
    plt.show()