"""
	Dynamic Program class - implements only backward dynamic programming.
	Ideal usage is to have an objective function defined by the user, however they'd like.

	The user defines as many StateVariable instances as they have state variables in their DP and they define
	a DecisionVariable. The objective function should be prepared to take arguments with keyword values for each of
	these, where the keyword is determined by the name attribute on each instance. It then returns a value.

	For situations with multiple state variables, we have reducers, which, prior to minimization or maximization
	will reduce the number of state variables to one so that we can get a single F* for each input scenario.
	This is a class Reducer with a defined interface, and can then be extended. For our levee problem, we have a
	ProbabilisticReducer which keeps track of the probability of each set of state variables and can collapse by
	a master variable. Probabilities must be provided by the user.

	# TODO: Make sure to check how we can make the decision variable properly interact with the state variable - thinking
	# of making sure that the decision variable adds to the correct state variable items
	# TODO: Missing plan for how we assign probabilities here - needs to be incorporated somewhere
	# TODO: Missing choice constraints (eg, minimim possible selection at stage ___ is ___)


	At that point, usage for the levee problem should be something like:

	.. code-block:: python

		import support  # user's extra code to support the objective function
		import dypy as dp

		objective_function = support.objective_function
		height_var = dp.StateVariable("height")
		flow_var = dp.StateVariable("peak_flow")
		variance_var = dp.StateVariable("variance")
		decision_var = dp.DecisionVariable()
		decision_var.related_state = height_var  # tell the decision variable which state it impacts

		dynamic_program = dp.DynamicProgram(objective_function=objective_function,
											state_variables=(height_var, flow_var, variance_var),
											decison_variable=decision_var)
		dynamic_program.optimize()  # runs the backward recursion
		dynamic_program.get_optimal_values()  # runs the forward method to obtain choices

"""

import logging
import itertools

import six
import numpy
import pandas

from .reducers import Reducer, ProbabilisticReducer, VariableReducer
from .variables import StateVariable, DecisionVariable
from . import support

log = logging.getLogger("dypy")

__author__ = "nickrsan"
__version__ = "0.0.3b"


MAXIMIZE = numpy.max
MINIMIZE = numpy.min


def default_objective(*args, **kwargs):
	"""
		A bare objective function that makes all cells 0 - provided as a default so we can build stages where someone
		might do a manual override, but the DP should check to make sure the objective isn't the default with all cells
		at 0 before solving.

	:param args:
	:param kwargs:
	:return:
	"""

	return 0


class DynamicProgram(object):
	"""
		This object actually runs the DP - doesn't currently support multiple decision variables or probabilities.

		Currently designed to only handle backward DPs.

		Attributes:
			DynamicProgram.stages	list of Stage objects

		:param objective_function: What function provides values to populate each cell in each stage? Not required if
									you build your own stages with their own values

		:param selection_constraints: Is there a minimum value that must be achieved in the selection?
				If so, this should be a list with the required quantity at each time step

		:param decision_variable: a DecisionVariable instance object

		:param discount_rate: give the discount rate in timestep_size units. Though timesteps don't need to be in years, think
			of the discount rate as applying per smallest possible timestep size, so if your timestep_size is 40, then
			your discount rate will be for one timestep in 40 and will compound across the 40 timesteps in the single stage.
			Defaults to 0, which results in no discounting, and discounting is only enabled automatically if you use the DiscountedSimplePrior, or another Prior object
			that handles discounting.

		:param calculation_function: What function are we using to evaluate? Basically, is this a maximization (benefit)
		 or minimization (costs) setup. Provide the function object for max or min. Provide the actual `min` or `max functions
		 (don't run it, just the name) or if convenient, use the shortcuts dp.MINIMIZE or dp.MAXIMIZE

		:param max_selections: Up to what value of the state variable can we use to make our decision? If not provided,
			all values of the state variable will be allowed

		:param prior: Which class should be used to handle priors (best values from future stages) - SimplePrior is best in many
			cases, but may not apply everywhere and will be slow for large tables. Just provide the class object, not an instance.
	"""

	def __init__(self, objective_function=default_objective, timestep_size=None, time_horizon=None, discount_rate=0, state_variables=None, max_selections=None, selection_constraints=None, decision_variable=None, calculation_function=None, prior=None):

		self.stages = []
		self.timestep_size = timestep_size
		self.time_horizon = time_horizon
		self.discount_rate = discount_rate
		self.max_selections = max_selections

		self._all_states = None
		self._all_states_df = None
		self._initial_states =None
		self._state_keys = None

		self.state_variables = list()
		if state_variables:
			for variable in state_variables:
				self.add_state_variable(variable, setup_state=False)
			self._setup_state_variables()  # set up the state variables for the DP once all are added

		# set up decision variables passed in
		if not decision_variable:
			self.decision_variable = None
		else:
			self.decision_variable = decision_variable

		# Calculation Function
		self.objective_function = objective_function
		self.calculation_function = calculation_function

		# Default Prior Handler
		self.prior_handler_class = prior

		if self.calculation_function not in (numpy.max, numpy.min, MAXIMIZE, MINIMIZE):
			raise ValueError("Calculation function must be either 'numpy.max', 'numpy.min' or one of the aliases in this package of dp.MAXIMIZE or dp.MINIMIZE")

		if self.calculation_function is numpy.min:
			self.index_calculation_function = numpy.argmin
		else:  # we can do else because only options are numpy.min and numpy.max, with ValueError raised above if it's not one of those
			self.index_calculation_function = numpy.argmax

		# make values that we use as bounds in our calculations - when maximizing, use a negative number, and when minimizing, get close to infinity
		# we use this for any routes through the DP that get excluded
		if self.calculation_function is numpy.max:
			self.exclusion_value = -9223372036854775808
		else:
			self.exclusion_value = 9223372036854775808  # max value for a signed 64 bit int - this should force it to not be selected in minimization

	def add_state_variable(self, variable, setup_state=True):
		"""

			Adds a state variable object to this dynamic program - automatically sets up the dynamic program
			to run with the new state variable when setup_state is True. Otherwise waits for you to call it manually.
			If you provide your state variables at DynamicProgram creation, then this method does not need to be called.

		:param variable: A StateVariable object - afterward, will be available in .state_variables
		:param setup_state: If True, runs _setup_state_variables after finishing. Default is True,
							but when adding a bunch of state variables, it's faster just to manually
							run it once at the end. It runs it by default so that for simple DPs, you
							don't need to think about it, but advanced, larger DPs, may wnat to set it
							to False and run _setup_state_variables once all are added
		:return:
		"""
		if not isinstance(variable, StateVariable):
			raise ValueError("Provided variable must be a StateVariable object. Can't add variable of type {} to DP".format(type(variable)))

		self.state_variables.append(variable)
		if setup_state:
			self._setup_state_variables()

	def _setup_state_variables(self):
		"""
			After adding state variables, performs setup operations for the DP
		:return:
		"""
		self._index_state_variables()  # make sure to reindex the variables when we add one
		self._all_states = list(itertools.product(*[var.values for var in self.state_variables]))
		self._all_states_df = pandas.DataFrame(self._all_states, columns=[var.variable_id for var in self.state_variables])
		self._setup_initial_state()
		self._state_keys = [var.variable_id for var in self.state_variables]  # will be in same order as provided to all_states

	def _setup_initial_state(self):
		var_values = []
		for var in self.state_variables:
			if var.initial_state is not None:  # if it specifies an initial value for this state, use it and only it
				var_values.append([var.initial_state])  # make it a list before appending - we'll use that in a moment
			else:
				var_values.append(var.values)
		self._initial_states = list(itertools.product(*var_values))

	def _index_state_variables(self):
		for index, variable in enumerate(self.state_variables):
			if not isinstance(variable, StateVariable):  # this is a bit silly to have this check twice, but this method checks it even if the user passes a list of StateVariables
				raise ValueError("Provided variable must be a StateVariable object. Can't add variable of type {} to DP".format(type(variable)))

			variable.column_index = index  # tell the variable what column it is

	def add_stage(self, name):
		"""
			Adds an empty stage to this DynamicProgram, tied to the currently set DecisionVariable and Prior classes
			associated with this DynamicProgram. Gives the stage the name provided as a parameter, usually generated
			automatically by the DP.

			This method is called automatically by default, but is exposed as a public method primarily so you can
			control its usage if you desire, or so you can override its behavior in a subclass (such as if you wish
			to manually handle Stage creation in order to control the process for a specific dynamic programming problem).

		:param name: The name associated with the stage. Generated by the name prefix and the stage ID when called via build_stages
		:return: None
		"""
		stage = Stage(name=name, decision_variable=self.decision_variable, parent_dp=self, prior_handler=self.prior_handler_class)

		self.stages.append(stage)
		self._index_stages()

	def _index_stages(self):

		# assigning .number allows us to have constraints on the number of items selected at each stage
		if len(self.stages) > 1:  # if there are at least two, then set the next and previous objects on the first and last
			self.stages[0].next_stage = self.stages[1]
			self.stages[0].number = 0
			self.stages[-1].previous_stage = self.stages[-2]
			self.stages[-1].number = len(self.stages) - 1

		for i, stage in enumerate(self.stages[1:-1]):  # for all stages except the first and last, then we set both next and previous
			i+=1  # need to add 1 or it won't work because we're slicing off the first item
			self.stages[i].next_stage = self.stages[i+1]
			self.stages[i].previous_stage = self.stages[i-1]
			self.stages[i].number = i

	def build_stages(self, name_prefix="Step"):
		"""
			Make a stage for every timestep

		:param name_prefix: The string that will go before the stage number when printing information
		:return:
		"""

		for stage_id in range(0, self.time_horizon+1, self.timestep_size):
			self.add_stage(name="{} {}".format(name_prefix, stage_id))

	def _is_default(self):
		"""
			Checks if we are using the default objective function *and* haven't modified any stage values. We don't want
			to solve the DP in this case, but instead raise an error to tell the user they're not using the solver
			appropriately

		:return:
		"""
		log.debug("checking that stages are not in default state")
		for stage in self.stages:
			if numpy.sum(stage.matrix) != 0:
				return False  # not in default state - have other values

		return True  # if we make it out here, we're in the default state

	def run(self):

		if self._is_default():
			raise ValueError("Objective function must be provided before running DP. Either provide it as an argument "
							 "at creation, or set the .objective_function parameter to a function object (*not* a function result)")

		if not self.decision_variable or len(self.state_variables) == 0:
			raise ValueError("Decision Variable and State Variables must be attached to DynamicProgram before running. Use .add_state_variable to attach additional state variables, or set .decision_variable to a DecisionVariable object first")

		# build a matrix where everything is 0  - need to figure out what the size of the x axis is
		# this matrix should just have a column for each timestep (we'll pull these out later), which will then be used by
		# each stage to actually build its own matrix
		rows = int(self.time_horizon/self.timestep_size)  # this is the wrong way to do this - the matrix should
		matrix = numpy.zeros((rows, ))

		for stage in self.stages:
			stage.build()

		# TODO: This needs to be updated for the more modern form - this probable is controlled by the stage instead
		#for stage in range(rows):
		#	for index, row in enumerate(matrix):
		#		matrix[index][stage] = support.present_value(self.objective_function(), index, year=stage*self.timestep_size, discount_rate=self.discount_rate )

		# This next section is old code from a prior simple DP - it will be removed, but was how the set of stages was
		# built previously so I can see what the usage was like while building this for multiple objectives

		# initiate the optimization and retrieval of the best values
		self.stages[-1].optimize()
		self.stages[0].get_optimal_values()


class Stage(object):
	def __init__(self, name, decision_variable, parent_dp, previous=None, next_stage=None, prior_handler=None):
		"""

		:param name:
		:param decision_variable: an iterable containing benefit or cost values at each choice step
		:param max_selections: How many total items are we selecting?
		:param previous: The previous stage, if one exists
		:param next_stage: The next stage, if one exists
		"""
		self.name = name
		self.parent_dp = parent_dp
		self.decision_variable = decision_variable
		self.prior_handler_class = prior_handler
		self.prior_handler = None
		self.create_prior_handler()

		self.decision_amount = None
		self.future_value_of_decision = None

		self.next_stage = next_stage
		self.previous_stage = previous
		self.matrix = None  # this will be created from the parameters when .optimize is run
		self.number = None

		self.pass_data = []
		self.choices_index = []

	def create_prior_handler(self):
		if self.prior_handler_class is not None:
			if isinstance(self.prior_handler_class, Prior):  # if it's already an instantiated object
				self.prior_handler = self.prior_handler_class  # set the item
				self.prior_handler.stage = self  # tell it that it applies to this item
				self.prior_handler_class = self.prior_handler.__class__  # and set the class
			else:
				self.prior_handler = self.prior_handler_class(stage=self)  # otherwise, create it

	def build(self):
		"""
			Builds the stage table - evaluates the objective function for each location, passing the various
			values to it so it can provide a result. Does *not* handle prior data though - that is handled when
			we actually run the DP. We separate this out so that people who want to intervene between generating
			the stage table and handling the prior can do so.

		:return:
		"""

		log.info("Building stage {}".format(self.number))
		if self.number == 0:
			states = self.parent_dp._initial_states  # use only the initial states if provided for the first stage
		else:
			states = self.parent_dp._all_states  # otherwise use all states for the future
		state_kwargs = self.parent_dp._state_keys
		decisions = self.decision_variable.values
		self.matrix = numpy.zeros((len(states), len(decisions)))

		for row_id, state_values in enumerate(states):
			for column_id, decision_value in enumerate(decisions):
				state_data = dict(zip(state_kwargs, state_values))  # build the kwarg pairs into a dictionary we can pass into the objective function
				decision_data = {self.decision_variable.variable_id: decision_value}
				objective_value = self.parent_dp.objective_function(stage=self,	**support.merge_dicts(state_data,
																										  decision_data))
				self.matrix[row_id][column_id] = objective_value  # have this on a separate line for debugging

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name

	def optimize(self, prior=None):
		"""
			This handles the actual backward DP calculation - assumes that the stage has a matrix that is built
			already with appropriate values. Calls the Prior object to apply future values back to the current stage,
			when necessary.

			* Need to think about where reducers fit in
			* Need to figure out how we want to handle both selection constraints, but also decision/state interactions
			Right now, we can say which state a decision feeds back on - but what's going through my head about that
			is:

			1. Do we need that if we expect an objective function? I think we do because that determines
			how we take values from the future and propogate them backward. This is where maybe we
			might not actually be able to use numpy broadcasting? Or maybe we have to shift the array
			we're broadcasting to align based on the best decision - not sure - need to think of how
			we're going to handle that a bit more

			2. What about where there's some limit imposed by decision var / state var interaction. For
			example, with the course studying test problem - in the last stage, the state variable
			can't be less than the decision variable. Maybe that's always the case though, and is
			only a last stage problem?

		:param prior:
		:return:
		"""

		#if self.parent_dp.selection_constraints and self.number is None:
		#	raise ValueError("Stage number(.number) must be identified in sequence in order to use selection constraints")

		#	# set up maximum selection constraints based on what will have been required previously
		#	if self.parent_dp.selection_constraints:  # then we have selections constraints
		#		for row_index, row_value in enumerate(self.matrix):
		#				if row_index >= len(self.matrix) - self.parent_dp.selection_constraints[self.number]:  # if the row is less than the constrained amount for this stage
		#					for column_index, column_value in enumerate(self.matrix[row_index]):
		#						self.matrix[row_index][column_index] = self.parent_dp.exclusion_value
		#
		#	# now calculate the remaining values

		if self.prior_handler is None:
			raise ValueError("Prior handler not initialized - make sure you specify which class handles priors either"
							 "at DynamicProgram initialization, or when building your stages. Set .prior_handler_class on either"
							 "one to the class object that will handle priors.")

		self.prior_handler.data = prior
		self.prior_handler.matrix = self.matrix
		# Add in previous stage
		if self.prior_handler.exists():  # if we have prior data
			self.matrix = self.prior_handler.apply()  # then apply it to the current matrix

		self.pass_data = self.parent_dp.calculation_function(self.matrix, axis=1)  # sum the rows and find the best
		self.choices_index = self.parent_dp.index_calculation_function(self.matrix, axis=1)  # get the column of the min/max value

		if self.previous_stage:
			self.previous_stage.optimize(self.pass_data)  # now run the prior stage

	def _filter_available_states(self):
		"""
			An important function that creates the matrix of available choices when
			we do the forward calculation - by default, goes through each state variable
			and subsets the matrix of values to the ones that are feasible based
			on the state variable's current state. If the variable has no current state,
			all options are evaluated. If it has a current states, then it is subset
			based on the variable's availability function, which defines the relationship
			of the current state to available choices. See documentation about variables
			for more.
		:return: None - sets self.search_matrix to available values and self.search_states to the corresponding state values
		"""
		if self.number == 0:  # in the first state, we've already filtered it
			self.search_matrix = self.matrix
			# TODO: Doesn't set the search states
			return

		self.search_matrix = self.matrix  # start with the full matrix
		self.search_states = self.parent_dp._all_states_df

		for variable in self.parent_dp.state_variables:
			if variable.current_state is None:  # skip any variable without a current state - we'll use all options then
				continue

			# this next line is a doozy - it gets the row ids where the *current* state variable's
			# current value has the relationship defined in the variable's availability function
			# - for example, if the current state of the current var is 2 and the availability
			# function is numpy.equal, then it gets the row indices where this state var equals 2
			index = self.search_states.index[variable.availability_function(self.search_states[variable.variable_id], variable.current_state)]

			self.search_states = self.search_states.loc[index]  # subset the search states - this will then be used for the next state var
			self.search_matrix = self.search_matrix[index]  # subset the search matrix to match  - this will be used by the later function to decide what's best

	def get_optimal_values(self, prior=0):
		"""
			After running the backward DP, moves forward and finds the best choices at each stage.
		:param prior: The value of the choice at each stage
		:return:
		"""

		# TODO: we'll want to implement the max_selections again using the available state filtering - might just be numpy.less
		#if self.parent_dp.max_selections:  # this format probably only works for 1 state variable and will need to be re-engineered
		#	max_selections = self.parent_dp.max_selections
		#else:
		#	max_selections = len(self.parent_dp._all_states)

		#amount_remaining = max_selections - prior
		self._filter_available_states()  # gives us self.search_matrix with available options

		if self.search_matrix.any():
			# start with all options

			self.best_option = self.parent_dp.calculation_function(self.search_matrix)
			coord_sets = numpy.argwhere(self.search_matrix == self.best_option)
			self.row_of_best = coord_sets[0][0]
			self.column_of_best = coord_sets[0][1]

		else:
			self.best_option = 0
			self.row_of_best = 0
			self.column_of_best = 0

		#if self.selection_constraints:
		#	number_of_items = max([0, column_of_best - self.selection_constraints[self.number]])  # take the max with 0 - if it's negative, it should be 0
		#else:
		self.decision_amount = self.parent_dp.decision_variable.values[self.column_of_best]
		self.future_value_of_decision = self.best_option
		log.info("{} - Decision Amount at Stage: {}, Total Cost/Benefit: {}".format(self.name, self.decision_amount, self.best_option))

		# TODO: Should make the value of each state at the time of decision in each stage accessible on stage objects
		if self.decision_variable.related_state is not None:  # update the state of the state variable to match the new decision
			if self.decision_variable.related_state.current_state is None:
				self.decision_variable.related_state.current_state = self.decision_amount
			else:
				self.decision_variable.related_state.current_state += self.decision_amount

		if self.next_stage:
			self.next_stage.get_optimal_values(prior=self.decision_amount + prior)


class Prior(object):
	"""
		The Prior classes provide different ways of applying future stage values to earlier stages. SimplePrior
		includes an implementation that will work for most single variable problems, but which may not work for
		multi-variable problems. This class can be subclassed and have the apply method overridden to provide
		a different implementation. The apply method should return the new matrix

		:param stage: The Stage object to attach this prior to - allows for different stages to have different prior
			application methods, if needed
		:param data: The values from the future stage to apply back to the previous stage
		:param matrix: The matrix (2D numpy array) for consideration in the current stage
		:param transformation: A function - can be used by subclasses to transform values prior to applying them

	"""

	def __init__(self, stage, data=None, matrix=None):
		self.data = data
		self.matrix = matrix
		self.parent_stage = stage

		self._transformation = lambda x: x  # default transformation just returns the value
		self._transformation_kwargs = dict()

		self.setUp()

	def setUp(self):
		pass

	def exists(self):
		if self.data is not None:
			return True
		return False

	def apply(self):
		raise NotImplementedError("Must use a subclass of Prior, not Prior class itself")


class SimplePrior(Prior):
	"""
		A simple implementation of the prior class - given a single set of optimal values from the future stage (as a 1D
		numpy array), it applies these values to the previous stage by flipping the axis of the decision
		and shifting the values down by 1 as we move across the decisions.

		This prior application method *won't* work for multiple state variables or for state variables with different
		discretization than decision variables. For more advanced situations with multiple states, we'll need to use
		the linked_state attribute on the decision variables to understand how to apply priors. This functionality
		may need to be re-engineered or expanded.
	"""

	def apply(self):
		for row_index, row_value in reversed(list(enumerate(self.matrix))):  # go from bottom to top because we modify the items a row below as we go, but need their valus to be intact, so starting at the bottom allows us to use clean values for calculations
			for column_index, column_value in reversed(list(enumerate(row_value))):
				if row_index - column_index <= 0:  # in these cases, we can't actually pull the prior value
					continue

				#if column_value == 0 and row_index - column_index >= 0:  # we only need to calculate fields that meet this condition - the way it's calculated, others will cause an IndexError anyway
				stage_value = self.matrix[row_index][column_index]  # the value for this stage is on 1:1 line, so, it is where the indices are equal

				if stage_value == self.parent_stage.parent_dp.exclusion_value:  # skip anything excluded elsewhere
					continue

				try:
					prior_value = self.data[row_index-column_index]
					prior_value = self._transformation(prior_value, **self._transformation_kwargs)
					self.matrix[row_index+1][column_index] = stage_value + prior_value
				except IndexError:
					continue
		return self.matrix


class DiscountedSimplePrior(SimplePrior):
	"""
		Same as SimplePrior, but discounts values from future stages before applying them to the current
		stage. This discounting is cumulative (as in stage n+2 will be discounted and applied to stage n+1
		and then that summed value will then be discounted and applied to stage n.

		Discount rate is taken from the DynamicProgram object and will be transformed (simply) to cover the timestep
		size on the DynamicProgram - see the DynamicProgram parameter documentation for more.
	"""

	def setUp(self):
		self._transformation = support.present_value
		self._transformation_kwargs = {'year': self.parent_stage.parent_dp.timestep_size,
									   'discount_rate': self.parent_stage.parent_dp.discount_rate}
