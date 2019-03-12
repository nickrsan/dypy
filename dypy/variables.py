import numpy

class StateVariable(object):
	"""
		Not sure what I'm going to do with this yet, but I think we'll need it in order to have rows for interactions
		between multiple state variables.

		When we go to use all the state variables together, we'll need to discretize them each, and then we'll need to combine
		them to get all the rows in the table for each stage. Assuming we have some attribute .discretized that contains
		all the values for this state variable and that the DynamicProgram class has a list of these state variables called
		.state_variables, we can get all possible combinations for generating a row using `itertools.product(*[var.discretized for var in self.state_variables])`
		Note the asterisk at the front, which takes that list and expands it so each one is an individual argument to
		itertools.product

		We can then use this by taking the name attribute of the state variable and passing it as the kwarg to the objective
		function along with the discretized value. So then we have a DP class that accepts a list of variables, an objective
		function, and then a preprocessor function that handles aggregation of choices between filling in the matrix
		and use of minimization (a function that accepts the array and reduces the choices so that we can actually minimize
		or maximize. We need this to reduce multiple state variables to a single state variable.
	"""

	def __init__(self, name, values):
		self.name = name
		self.values = values

		self.column_index = None  # this will be set by the calling DP - it indicates what column in the table has this information


class DecisionVariable(object):
	def __init__(self, name, related_state=None, minimum=None, maximum=None, step_size=None, options=None):
		"""
			We'll use this to manage the decision variable - we'll need columns for each potential value here
		:param name:
		:param state: the StateVariable object that this DecisionVariable directly feeds back on
		"""
		self.name = name
		self.related_state = related_state

		self._min = minimum
		self._max = maximum
		self._step_size = step_size
		self._options = options
		if options:
			self._user_set_options = True  # keep track so we can zero it out later if they set min/max/stepsize params
		else:
			self._user_set_options = False

		self.constraints = {}

	# we have all of these simple things as @property methods instead of simple attributes so we can
	# make sure to have the correct behaviors if users set the options themselves
	@property
	def minimum(self):
		return self._min

	@property
	def maximum(self):
		return self._max

	@property
	def step_size(self):
		return self._step_size

	@minimum.setter
	def minimum(self, value):
		self._min = value
		self._reset_options()

	@maximum.setter
	def maximum(self, value):
		self._max = value
		self._reset_options()

	@step_size.setter
	def step_size(self, value):
		self._step_size = value
		self._reset_options()

	def _reset_options(self):
		if not self._user_set_options:
			self._options = None  # if we change any of the params, clear the options

	@property
	def options(self):
		if self._options:  # if they gave us options
			return self._options
		elif self._min and self._max and self.step_size:
			if type(self._min) == "int" and type(self._max) == "int" and type(self.step_size) == "int":  # if they're all integers we'll use range
				self._options = range(self._min, self._max, self.step_size)  # cache it so next time we don't have to calculate
			else:
				# the `num` param here just transforms step_size to its equivalent number of steps for linspace. Add 1 to capture accurate spacing with both start and endpoints
				self._options = numpy.linspace(start=self._min, stop=self._max, num=int((self._max-self._min)/self.step_size)+1, endpoint=True)

			self._user_set_options = False
			return self._options

		raise ValueError("Can't get DecisionVariable options - need either explicit options (.options) or a minimum value, a maximum value, and a step size")

	@options.setter
	def options(self, value):
		self._options = value
		self._user_set_options = True

	def add_constraint(self, stage, value):
		"""
			Want to figure out a way here to store also whether this constraint is a minimum or a maximum value constraint.
			Need to think how we'd handle that behavior
		:param stage:
		:param value:
		:return:
		"""
		pass
