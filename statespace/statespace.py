import numpy as np


class LinearSDEStateSpace:

    def __init__(self, initial_state, model_drift, model_mean, model_covar, model_ext_covar, model_noise_matrix,
                 driving_process,
                 observation_matrix, modelCase=1, truncation_level=1e-8, rng=np.random.default_rng(), theta=-1.):
        """
        - model drift = e^A(t-u)
        - model mean = e^A(t-u) h
        - model covar = e^A(t-u) h hT e^A(t-u)T
        - model noise matrix = B s.t. Xt = drift @ Xs + B @ noise
        """
        # need to place checks on the dimensions of the return values of these functors
        self.state = initial_state
        self.drift = model_drift
        self.mean = model_mean
        self.covar = model_covar
        self.ext_covar = model_ext_covar
        self.B = model_noise_matrix

        self.driving = driving_process
        self.mu_W = self.driving.get_mu_W()
        self.var_W = self.driving.get_var_W()
        self.noise_model = modelCase

        self.truncation = truncation_level

        self.H = observation_matrix

        self.theta = theta

        self.rng = rng

    def get_model_drift(self, interval):
        return self.drift(interval)

    def get_model_m(self, interval, jump_times, jump_sizes):
        # m should contain NOT mu_W if marginalising wrt mu_W
        return np.atleast_2d((self.mu_W * jump_sizes * self.mean(interval, jump_times)).sum(axis=-1)).T

    def get_model_S(self, interval, jump_times, jump_sizes):
        return (self.var_W * jump_sizes * self.covar(interval, jump_times)).sum(axis=-1)

    def get_model_Ce(self, interval):
        cov_constant = self.driving.subordinator.small_jump_covariance(truncation=self.truncation,
                                                                       case=self.noise_model)
        return (self.var_W * cov_constant[0] + (self.mu_W ** 2) * cov_constant[1]) * self.ext_covar(interval)

    def get_model_B(self):
        return self.B

    def get_model_H(self):
        return self.H

    def get_model_var_W(self):
        return self.var_W

    def getModelCase(self):
        return self.noise_model

    def set_state(self, state):
        self.state = state

    def get_driving_jumps(self, rate, M=2000, gamma_0=0.):
        return self.driving.subordinator.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0, truncation=self.truncation)

    def increment_state(self, interval, M=2000, gamma_0=0.):
        jump_times, jump_sizes = self.get_driving_jumps(rate=1. / interval, M=M, gamma_0=gamma_0)
        m_vec = self.get_model_m(interval=interval, jump_times=jump_times, jump_sizes=jump_sizes)
        S_mat = self.get_model_S(interval=interval, jump_times=jump_times, jump_sizes=jump_sizes)
        Ce = self.get_model_Ce(interval)
        try:
            C_mat = np.linalg.cholesky(S_mat + Ce)
            e = np.atleast_2d(m_vec + (C_mat @ np.atleast_2d(self.rng.normal(size=S_mat.shape[1])).T))
        except np.linalg.LinAlgError:
            e = np.atleast_2d(np.zeros(S_mat.shape[1])).T

        new_state = self.drift(interval) @ self.state + self.B @ e
        return new_state, m_vec, S_mat + Ce

    def observe_in_noise(self, kv):
        return (self.H @ self.state + np.sqrt(self.var_W * kv) * self.rng.normal()).item()

    def generate_observations(self, times, kv):
        intervals = np.diff(times)
        """ Note state is 0 initially, so initialising according to normal random variable with 0 mean """
        observations = [self.observe_in_noise(kv)]
        positions = [self.observe_in_noise(0.)]
        trends = [self.observe_in_noise(0.)]
        ms = []
        Ss = []
        for diff in intervals:
            self.state, m, S = self.increment_state(diff)
            ms.append(m)
            Ss.append(S)
            observations.append(self.observe_in_noise(kv))
            positions.append(self.state[0,0])
            trends.append(self.state[1, 0])
        return [observations, positions, trends], ms, Ss


class LangevinStateSpace(LinearSDEStateSpace):

    def __init__(self, initial_state, theta, driving_process, observation_matrix, modelCase=1, truncation_level=1e-6,
                 rng=np.random.default_rng()):
        self.theta = theta
        self.P = 2
        super().__init__(initial_state, self.langevin_drift, self.langevin_mean, self.langevin_covar,
                         self.langevin_ext_covar,
                         self.langevin_noise_matrix, driving_process, observation_matrix, modelCase=modelCase,
                         truncation_level=truncation_level, rng=rng, theta=theta)

    # model specific functors
    langevin_drift = lambda self, interval: np.exp(self.theta * interval) * np.array(
        [[0., 1. / self.theta], [0., 1.]]) + np.array([[1., -1. / self.theta], [0., 0.]])

    langevin_mean = lambda self, interval, jtime: np.exp(self.theta * (interval - jtime)) * np.atleast_2d(
        np.array([[1. / self.theta, 1.]])).T + np.atleast_2d(np.array([[-1. / self.theta, 0.]])).T

    langevin_covar = lambda self, interval, jtime: np.exp(2. * self.theta * (interval - jtime)) * np.array(
        [[1. / (self.theta ** 2), 1. / self.theta], [1. / self.theta, 1.]])[:, :, np.newaxis] + np.exp(
        self.theta * (interval - jtime)) * np.array(
        [[-2. / (self.theta ** 2), -1. / self.theta], [-1. / self.theta, 0.]])[:, :, np.newaxis] + np.array(
        [[1. / (self.theta ** 2), 0.], [0., 0.]])[:, :, np.newaxis]

    langevin_ext_covar = lambda self, interval: (np.exp(2. * self.theta * interval) - 1.0) * np.array(
        [[0.5 / (self.theta ** 3), 0.5 / (self.theta ** 2)], [0.5 / (self.theta ** 2), 0.5 / self.theta]]).reshape(
        (self.P, self.P)) + (np.exp(self.theta * interval) - 1.0) * np.array(
        [[-2. / (self.theta ** 3), -1. / (self.theta ** 2)], [-1. / (self.theta ** 2), 0.]]).reshape(
        (self.P, self.P)) + interval * np.array([[1 / (self.theta ** 2), 0], [0, 0]]).reshape((self.P, self.P))

    langevin_noise_matrix = np.eye(2)
