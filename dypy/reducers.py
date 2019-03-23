import logging

log = logging.getLogger("dypy.reducers")


class Reducer(object):
	"""
		Reduces multiple state variables to a single state variable so we can just minimize
	"""
	def __init__(self, variable, stage):
		self.variable = variable  # reference to StateVariable object
		self.stage = stage  # reference to Stage object


class VariableReducer(Reducer):
	"""
		Given a StateVariable, reduces the table size by collapsing all other variables - can do this by min/max/mean/sum
		of all options.

		Saving implementation here until after we have a better sense for how the rest of this will be implemented'

		:param variable:
		:param stage:
	"""
	pass

class ProbabilisticReducer(Reducer):
	"""
		Given a StateVariable to process (S), and a set of StateVariables to hold constant (Cs), reduces S for each
		combination of Cs by multiplying the objective values in the records for S by their probabilities and summing them.

		We should be able to actually just make this have a single column for probabilities so that we can do the same
		thing we planned to do for the variable reducer and just (ignoring the first paragraph of this docstring)
		select a master variable, get all rows for it, multiply those rows by the probability field, and sum them up.
	"""
	pass
