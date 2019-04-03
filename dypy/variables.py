import re
import logging

import six

import numpy

VARIABLE_ID_VALIDATION_PATTERN = re.compile('[^a-zA-Z0-9_]|_')
VARIABLE_NUMERAL_BEGINNING_PATTERN = re.compile('^[0-9]')

log = logging.getLogger("dypy.variables")


class AbstractVariable(object):
	def __init__(self, name, variable_id=None, minimum=None, maximum=None, step_size=None, values=None, *args, **kwargs):
		self.name = name
		self.variable_id = check_variable_id(name=name, variable_id=variable_id)

		self._min = minimum
		self._max = maximum
		self._step_size = step_size
		self._options = values
		if values is not None:
			self._user_set_options = True  # keep track so we can zero it out later if they set min/max/stepsize params
		else:
			self._user_set_options = False



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
	def values(self):
		if self._options is not None:  # if they gave us options or we already made them
			return self._options
		elif self._min is not None and self._max is not None and self._step_size is not None:
			if type(self._min) == "int" and type(self._max) == "int" and type(self.step_size) == "int":  # if they're all integers we'll use range
				self._options = range(self._min, self._max, self.step_size)  # cache it so next time we don't have to calculate
			else:
				# the `num` param here just transforms step_size to its equivalent number of steps for linspace. Add 1 to capture accurate spacing with both start and endpoints
				self._options = numpy.linspace(start=self._min, stop=self._max, num=int((self._max-self._min)/self.step_size)+1, endpoint=True)

			self._user_set_options = False
			return self._options

		raise ValueError("Can't get DecisionVariable options - need either explicit options (.options) or a minimum value, a maximum value, and a step size")

	@values.setter
	def values(self, value):
		self._options = value
		self._user_set_options = True


class StateVariable(AbstractVariable):
	"""

		:param name:
		:param values:
		:param initial_state: the initial value of this state variable at stage 0 - used when getting the ultimate
						solution - if not provided or None, then any state can be usd in the first stage, which is often
						not desired.
		:param availability_function: A numpy function indicating which states in the next stage are valid selections given
						the value in the current stage plus decisions. Should be:
						- numpy.equal (default) - only values that match the current state of this variable are available selections
						- numpy.not_equal - only values *not* matching the current state are valid
						- numpy.greater - only state values greater than the current state are available selections
						- numpy.greater_equal - same as above, but greater than or equal
						- numpy.less - only state values less than the current state are available selections
						- numpy.less_equal - same as above, but less than or equal

						Any function that takes a 2D numpy array as parameter one and the state value as parameter 2 and returns
						a new 2D array is valid.
		:param variable_id: will be used as the kwarg name when passing the value of the state into the objective function.
				If not provided, is generated from name by removing nonalphanumeric or underscore characters, lowercasing,
				and removing numbers from the beginning. If it is provided, it is still validated into a Python kwarg
				by removing leading numbers and removing non-alphanumeric/underscore characters, while leaving any capitalization
				intact
	"""

	def __init__(self, *args, **kwargs):

		self.column_index = None  # this will be set by the calling DP - it indicates what column in the table has this information

		if 'initial_state' in kwargs:
			self.initial_state = kwargs['initial_state']
		else:
			self.initial_state = None

		if 'availability_function' in kwargs:
			self.availability_function = kwargs['availability_function']
		else:
			self.availability_function = numpy.equal

		self.current_state = self.initial_state

		if six.PY3:
			super().__init__(*args, **kwargs)
		elif six.PY2:
			super(StateVariable, self).__init__(*args, **kwargs)


class DecisionVariable(AbstractVariable):
	"""
		We'll use this to manage the decision variable - we'll need columns for each potential value here

	:param name:
	:param related_state: the StateVariable object that this DecisionVariable directly feeds back on
	:param variable_id: will be used as the kwarg name when passing the value of the state into the objective function.
			If not provided, is generated from name by removing nonalphanumeric or underscore characters, lowercasing,
			and removing numbers from the beginning. If it is provided, it is still validated into a Python kwarg
			by removing leading numbers and removing non-alphanumeric/underscore characters, while leaving any capitalization
			intact
	"""

	def __init__(self, *args, **kwargs):
		if 'related_state' in kwargs:
			self.related_state = kwargs['related_state']
		else:
			self.related_state = None

		self.constraints = {}

		if six.PY3:
			super().__init__(*args, **kwargs)
		elif six.PY2:
			super(DecisionVariable, self).__init__(*args, **kwargs)

	def add_constraint(self, stage, value):
		"""
			Will be used to add constraints on how much or little of the decision variable is chosen in each stage.
			Not yet implemented

			Want to figure out a way here to store also whether this constraint is a minimum or a maximum value constraint.
			Need to think how we'd handle that behavior

		:param stage:
		:param value:
		:return:
		"""
		raise NotImplementedError("Decision constraints aren't yet implemented. Sorry!")


def check_variable_id(name, variable_id):
	"""
		Given a full variable name and a current variable_id returns the keyword argument name for the variable.
		Designed for private within-package usage, but public for inspection and overriding.

	:param name: Full name of a variable
	:param variable_id: the current variable ID
	:return: sanitized new variable_id, suitable for usage in a Python keyword argument
	"""
	if variable_id is None:
		variable_id = name.lower()

	# replace non-alphanumeric values with _
	variable_id = re.sub(VARIABLE_ID_VALIDATION_PATTERN, '_', variable_id)
	# strip numbers off the beginning
	variable_id = re.sub(VARIABLE_NUMERAL_BEGINNING_PATTERN, '', variable_id)

	return variable_id
