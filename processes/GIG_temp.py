class GIGProcess(__JumpLevyProcess):

	def __init__(self, delta=None, gamma=None, lambd=None, rng=np.random.default_rng()):
		self.set_parameters(delta, gamma, lambd)
		super().__init__(rng=rng)


	def set_parameters(self, delta, gamma, lambd):
		self.delta = delta
        self.gamma = gamma
        self.lambd = lambd


    def get_parameters(self):
        return {"delta": self.delta, "gamma": self.gamma, "lambd": self.lambd}


    def simualate_jumps(self, rate=1., M=100, gamma_0=0.):
    	if np.abs(self.lambd) >= 0.5:
    		simulator = self.__SimpleSimulator(self, rng=self.rng)
    		jtimes, jsizes = simulator.simulate_internal_jumps(rate, M, gamma_0)
    	else:
    		simulator1 = self.__N1(self, rng=self.rng)
    		simulator2 = self.__N2(self, rng=self.rng)
    		jtimes1, jsizes1 = simulator1.simulate_internal_jumps(rate, M, gamma_0)
    		jtimes2, jsizes2 = simulator2.simulate_internal_jumps(rate, M, gamma_0)
    		jtimes = np.append(jtimes1, jtimes2)
    		jsizes = np.append(jsizes1, jsizes2)

    	if self.lambd > 0:
    		# simulate gamma component
    		pass
    	return jtimes, jsizes


    class __SimpleSimulator(__JumpLevyProcess):
    	def __init__(self, outer: GIGProcess, rng=np.random.default_rng()):
    		self.outer = outer
			super().__init__(rng=rng)
			self.tsp_generator = TemperedStableProcess(alpha=0.5, beta=outer.gamma**2, C=outer.delta*gammafnc(0.5)/(np.sqrt(2.)*np.pi), rng=outer.rng)

		def __generate_z(self, x):
    		# i think shape of z is implicit
			return np.sqrt(self.rng.gamma(shape=0.5, scale=(2.*self.outer.delta**2)/x))

		def thinning_func(self, z):
			return 2. / (np.pi * z * hankel_squared(np.abs(self.outer.lambd)), z)

		def accept_reject_simulation(self, x, z, thinning_func, rate):
			assert(x.shape == z.shape)
			acceptance_seq = thinning_func(z)
			u = self.rng.uniform(low=0.0, high=1.0, size=x.size)
	        x_acc = x[u < acceptance_seq]
	        times = self.rng.uniform(low=0.0, high=1./rate, size=x_acc.size)
	        return times, x_acc

		def simulate_internal_jumps(self, rate=1.0, M=100, gamma_0=0.):
			_, x = self.tsp_generator.simulate_jumps(rate, M, gamma_0)
			z = self.__generate_z(x)
			jtimes, jsizes = self.accept_reject_simulation(x, z, thinning_func=self.thinning_func, rate=rate)
			return jtimes, jsizes


    class __N1(__JumpLevyProcess):
    	def __init__(self):
    		pass

    class __N2(__JumpLevyProcess):
    	def __init__(self):
    		pass