import numpy as np

class LinearSDEStateSpace:

	def __init__(initial_state, model_drift, model_mean, model_covar, driving_process):
		"""
		- model drift = e^A(t-u)
		- model mean = e^A(t-u) h
		- model covar = e^A(t-u) h hT e^A(t-u)T
		"""
		pass

