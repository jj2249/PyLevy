import numpy as np
import sys
from PyLevy.utils.maths_functions import psi, dpsi, hankel_squared, gammafnc


class __LevyProcess:
    """
	Base class for all Levy processes
	"""

    @staticmethod
    def integrate(evaluation_points, t_series, x_series, drift=0.):
        """
		Static method for plotting paths on a discretised time axis
		"""
        W = [x_series[t_series < point].sum() + drift * point for point in evaluation_points]
        return np.array(W).T


class __JumpLevyProcess(__LevyProcess):
    """
	Specific class for handling pure jump processes
	"""

    def __init__(self, rng=np.random.default_rng()):
        self.rng = rng

    def accept_reject_simulation(self, h_func, thinning_func, rate, M, gamma_0):
        """
		Simulate jump sizes and times using poisson epochs, a jump function and a thinning function
		"""
        epoch_seq = self.rng.exponential(scale=rate, size=M)
        epoch_seq[0] += gamma_0
        epoch_seq = epoch_seq.cumsum()
        x_seq = h_func(epoch_seq)
        acceptance_seq = thinning_func(x_seq)
        u = self.rng.uniform(low=0.0, high=1.0, size=x_seq.size)
        x_seq = x_seq[u < acceptance_seq]
        times = self.rng.uniform(low=0.0, high=1./rate, size=x_seq.size)
        return times, x_seq



class GammaProcess(__JumpLevyProcess):
    """
	Pure jump Gamma process
	"""

    def __init__(self, beta=None, C=None, rng=np.random.default_rng()):
        self.set_parameters(beta, C)
        super().__init__(rng=rng)

    def set_parameters(self, beta, C):
        self.beta = beta
        self.C = C

    def get_parameters(self):
        return {"beta": self.beta, "C": self.C}

    def h_func(self, epoch):
        return 1. / (self.beta * (np.exp(epoch / self.C) - 1.))

    def thinning_func(self, x):
        return (1. + self.beta * x) * np.exp(-self.beta * x)

    def simulate_jumps(self, rate=1.0, M=100, gamma_0=0.0):
        return self.accept_reject_simulation(self.h_func, self.thinning_func, rate, M, gamma_0)

    def unit_expected_residual_gamma(self, c):
        return (self.C / self.beta) * incgammal(1., 1. / (np.exp(c / self.C) - 1.))

    def unit_variance_residual_gamma(self, c):
        return (self.C / self.beta ** 2) * incgammal(2., 1. / (np.exp(c / self.C) - 1.))


class TemperedStableProcess(__JumpLevyProcess):

    def __init__(self, alpha=None, beta=None, C=None, rng=np.random.default_rng()):
        self.set_parameters(alpha, beta, C)
        super().__init__(rng=rng)

    def set_parameters(self, alpha, beta, C):
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def get_parameters(self):
        return {"alpha": self.alpha, "beta": self.beta, "C": self.C}

    def h_func(self, epoch):
        return np.power((self.alpha / self.C) * epoch, np.divide(-1., self.alpha))

    def thinning_func(self, x):
        return np.exp(-self.beta * x)

    def simulate_jumps(self, rate=1.0, M=100, gamma_0=0.0):
        return self.accept_reject_simulation(self.h_func, self.thinning_func, rate, M, gamma_0)

    def unit_expected_residual_tempered_stable(self, c):
        return (self.C * self.beta ** (self.alpha - 1.)) * incgammal(1. - self.alpha,
                                                                     self.beta * (self.alpha * c / self.C) ** (
                                                                             -1. / self.alpha))

    def unit_variance_residual_tempered_stable(self, c):
        return (self.C * self.beta ** (self.alpha - 2.)) * incgammal(2. - self.alpha,
                                                                     self.beta * (self.alpha * c / self.C) ** (
                                                                             -1. / self.alpha))


class GIGSubordinator(__JumpLevyProcess):

    def __init__(self, delta=None, gamma=None, lambd=None, rng=np.random.default_rng()):
        self.set_parameters(delta, gamma, lambd)
        super().__init__(rng=rng)

    def set_parameters(self, delta, gamma, lambd):
        self.delta = delta
        self.gamma = gamma
        self.lambd = lambd

    def get_parameters(self):
        return {"delta": self.delta, "gamma": self.gamma, "lambd": self.lambd}

    @staticmethod
    def __g(x, sd, td, f1, f2):
        a = 0
        b = 0
        c = 0
        if (x >= -sd) and (x <= td):
            a = 1
        elif (x > td):
            b = f1
        elif (x < -sd):
            c = f2
        return a + b + c

    def __generate_gamma_jumps(self, C, beta):
        epochs = self.generate_epochs()
        x = 1 / (beta * (np.exp(epochs / C) - 1))
        prob_acc = (1 + beta * x) * np.exp(-beta * x)
        return self.rejection_sampling(prob_acc, x)

    def __generate_tempered_stable_jumps(self, alpha, beta, delta):
        epochs = self.generate_epochs()
        C = (self.get_maxT() - self.get_minT()) * delta * gammafnc(0.5) / (np.sqrt(2) * np.pi)
        x = ((alpha * epochs) / C) ** (-1 / alpha)
        prob_acc = np.exp(-beta * x)
        return self.rejection_sampling(prob_acc, x)

    def __GIG_gamma_component(self):
        return self.__generate_gamma_jumps(max(0.0, self.lambd), self.gamma ** 2 / 2)

    def __generate_N1(self, z1, H0, lambd, delta, gamma_param):
        # Generate gamma process
        beta = 0.5 * gamma_param ** 2
        C = z1 / (np.pi * np.pi * np.absolute(lambd) * H0)  # Shape parameter of process at t = 1
        jump_sizes = self.__generate_gamma_jumps(C, beta, )

        """ Rejection sampling from Algorithm 6 """
        const1 = (z1 ** 2) * jump_sizes / (2 * delta ** 2)
        GIG_prob_acc = np.absolute(lambd) * gammafnc(np.abs(lambd)) * gammainc(np.abs(lambd), const1) / (
                ((z1 ** 2) * jump_sizes / (2 * delta ** 2)) ** np.abs(lambd))
        u = np.random.uniform(0., 1., size=jump_sizes.size)
        jump_sizes = jump_sizes[(u < GIG_prob_acc)]

        """ Sample from truncated Nakagami """
        C1 = np.random.uniform(0., 1., size=jump_sizes.size)
        l = C1 * gammainc(np.absolute(lambd), (z1 ** 2 * jump_sizes) / (2 * delta ** 2))
        zs = np.sqrt(((2 * delta ** 2) / jump_sizes) * gammaincinv(np.absolute(lambd), l))

        """ Thinning for process N1 """
        u = np.random.uniform(0., 1., size=jump_sizes.size)
        N1_prob_acc = H0 / (hankel_squared(np.abs(lambd), zs) *
                            (zs ** (2 * np.abs(lambd))) / (z1 ** (2 * np.abs(lambd) - 1)))
        jump_sizes = jump_sizes[(u < N1_prob_acc)]
        return jump_sizes

    def __generate_N2(self, z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon):
        """Generate point process N2 """

        """ Generate Tempered Stable Jump Size samples """
        alpha = 0.5
        beta = (gamma_param ** 2) / 2
        C = np.sqrt(2 * delta ** 2) * gammafnc(0.5) / ((np.pi ** 2) * H0)
        epochs = np.cumsum(
            np.random.exponential(1, N_epochs)) / T_horizon
        x = ((alpha * epochs) / C) ** (-1 / alpha)
        prob_acc = np.exp(-beta * x)
        u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
        jump_sizes = x[(u < prob_acc)]

        """ Rejection sampling based on Algorithm 7 """
        GIG_prob_acc = gammaincc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2))
        u = np.random.uniform(0., 1., size=jump_sizes.size)
        jump_sizes = jump_sizes[(u < GIG_prob_acc)]

        # Simulate Truncated Square-Root Gamma Process:
        C2 = np.random.uniform(low=0.0, high=1.0, size=jump_sizes.size)
        zs = np.sqrt(
            ((2 * delta ** 2) / jump_sizes) * gammaincinv(0.5,
                                                          C2 * (
                                                              gammaincc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2)))
                                                          + gammainc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2))))

        """Thinning for process N2"""
        u = np.random.uniform(0., 1., size=jump_sizes.size)
        N2_prob_acc = H0 / (zs * hankel_squared(np.abs(lambd), zs))
        jump_sizes = jump_sizes[(u < N2_prob_acc)]
        return jump_sizes


    def __GIG_harder_jumps(self):
        delta = self.delta
        gamma_param = self.gamma
        lambd = self.lambd
        N_epochs = self.get_epoch_number()
        T_horizon = self.get_maxT() - self.get_minT()
        a = np.pi * np.power(2.0, (1.0 - 2.0 * np.abs(lambd)))
        b = gammafnc(np.abs(lambd)) ** 2
        c = 1 / (1 - 2 * np.abs(lambd))
        z1 = (a / b) ** c
        H0 = z1 * hankel_squared(np.abs(self.lambd), z1)
        N1 = self.__generate_N1(z1, H0, lambd, delta, gamma_param)
        N2 = self.__generate_N2(z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon)
        jump_sizes = np.append(N1, N2)
        return jump_sizes

    def generate_jumps(self, epochs=None):
        """ Function does NOT use epochs """
        if np.abs(self.__lambd) >= 0.5:
            jumps = self.__GIG_simple_jumps()
        else:
            jumps = self.__GIG_harder_jumps()
        if self.__lambd > 0:
            p2 = self.__GIG_gamma_component()
            jumps = np.append(jumps, p2)
        super().set_jump_sizes(jumps)

    def marginal_samples(self, numSamples, tHorizon):
        """ Code is translated from MATLAB Code from:
			Jan Patrick Hartkopf (2022).
			gigrnd (https://www.mathworks.com/matlabcentral/fileexchange/78805-gigrnd),
			MATLAB Central File Exchange.
			Setup - - we sample from the two parameter version of the GIG(alpha, omega) where:
			P, a, b = lambd, gamma_param ** 2, delta ** 2,
		"""

        """ Which parameter is scaled by TIME ??? """

        a = self.gamma ** 2
        b = self.delta ** 2
        lambd = self.lambd
        omega = np.sqrt(a * b)
        swap = False
        if lambd < 0:
            lambd = lambd * -1
            swap = True
        alpha = np.sqrt(omega ** 2 + lambd ** 2) - lambd
        x = -psi(1, alpha, lambd)  # TODO CHECK
        if (x >= 0.5) and (x <= 2):
            t = 1
        elif x > 2:
            t = np.sqrt(2 / (alpha + lambd))
        elif x < 0.5:
            t = np.log(4 / (alpha + 2 * lambd))

        x = -psi(-1, alpha, lambd)  # TODO CHECK
        if (x >= 0.5) and (x <= 2):
            s = 1
        elif x > 2:
            s = np.sqrt(4 / (alpha * np.cosh(1) + lambd))
        elif x < 0.5:
            s = min(1 / lambd, np.log(1 + 1 / alpha + np.sqrt(1 / alpha ** 2 + 2 / alpha)))

        eta = -psi(t, alpha, lambd)
        zeta = -dpsi(t, alpha, lambd)
        theta = -psi(-s, alpha, lambd)
        xi = dpsi(-s, alpha, lambd)
        p = 1 / xi
        r = 1 / zeta
        td = t - r * eta
        sd = s - p * theta
        q = td + sd

        X = [0 for _ in range(numSamples)]
        for i in range(numSamples):
            done = False
            while not done:
                U = np.random.uniform(0., 1., size=1)
                V = np.random.uniform(0., 1., size=1)
                W = np.random.uniform(0., 1., size=1)
                if U < (q / (p + q + r)):
                    X[i] = -sd + q * V
                elif U < ((q + r) / (p + q + r)):
                    X[i] = td - r * np.log(V)
                else:
                    X[i] = -sd + p * np.log(V)
                f1 = np.exp(-eta - zeta * (X[i] - t))
                f2 = np.exp(-theta + xi * (X[i] + s))
                if (W * self.__g(X[i], sd, td, f1, f2)) <= np.exp(psi(X[i], alpha, lambd)):
                    done = True
        X = np.exp(X) * (lambd / omega + np.sqrt(1 + (lambd / omega) ** 2))
        if swap:
            X = 1 / X
        X = X / np.sqrt(a / b)
        X = X.reshape((1, X.shape[0]))
        return X[0]
