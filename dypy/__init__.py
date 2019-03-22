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

	At that point, usage for the levee problem should be something like:

	```
	import support  # user's extra code to support the objective function
	import dypy as dp

	objective_function = support.objective_function
	height_var = dp.StateVariable("height")
	flow_var = dp.StateVariable("peak_flow")
	variance_var = dp.StateVariable("variance")
	decision_var = dp.DecisionVariable()
	decision_var.related_state = height_var  # tell the decision variable which state it impacts

	# TODO: Make sure to check how we can make the decision variable properly interact with the state variable - thinking
			# of making sure that the decision variable adds to the correct state variable items
	# TODO: Missing plan for how we assign probabilities here - needs to be incorporated somewhere
	# TODO: Missing choice constraints (eg, min selection at stage ___ is ___)

	dynamic_program = dp.DynamicProgram(objective_function=objective_function,
										state_variables=(height_var, flow_var, variance_var),
										decison_variable=decision_var)
	dynamic_program.optimize()  # runs the backward recursion
	dynamic_program.get_optimal_values()  # runs the forward method to obtain choices
	```
"""

import logging
import itertools

import numpy

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

		Currently designed to only handle backward DPs
	"""

	def __init__(self, objective_function=default_objective, timestep_size=None, time_horizon=None, discount_rate=0, state_variables=None, max_selections=None, selection_constraints=None, decision_variable=None, calculation_function=None, prior=None):
		"""

		:param objective_function: What function provides values to populate each cell in each stage? Not required if
									you build your own stages with their own values

		:param selection_constraints: Is there a minimum value that must be achieved in the selection?
				If so, this should be a list with the required quantity at each time step

		:param decision_variables: list of DecisionVariable objects

		:param discount_rate: give the discount rate in "annual" units. Though timesteps don't need to be in years, think
			of the discount rate as applying per smallest possible timestep size, so if your timestep_size is 40, then
			your discount rate will be transformed to cover 40 timesteps (compounding).

		:param calculation_function: What function are we using to evaluate? Basically, is this a maximization (benefit)
		 or minimization (costs) setup. Provide the function object for max or min. Provide the actual `min` or `max functions
		 (don't run it, just the name) or if convenient, use the shortcuts dp.MINIMIZE or dp.MAXIMIZE

		:param max_selections: Up to what value of the state variable can we use to make our decision? If not provided,
			all values of the state variable will be allowed

		:param prior: Which class should be used to handle priors (best values from future stages) - SimplePrior is best in many
			cases, but may not apply everywhere and will be slow for large tables. Just provide the class object, not an instance.
		"""
		self.stages = []
		self.timestep_size = timestep_size
		self.time_horizon = time_horizon
		self.discount_rate = discount_rate
		self.max_selections = max_selections

		self._all_states = None
		self._state_keys = None
		if not state_variables:
			self.state_variables = []
		else:
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
		self._state_keys = [var.variable_id for var in self.state_variables]  # will be in same order as provided to all_states

	def _index_state_variables(self):
		for index, variable in enumerate(self.state_variables):
			if not isinstance(variable, StateVariable):  # this is a bit silly to have this check twice, but this method checks it even if the user passes a list of StateVariables
				raise ValueError("Provided variable must be a StateVariable object. Can't add variable of type {} to DP".format(type(variable)))

			variable.column_index = index  # tell the variable what column it is

	def add_stage(self, name):
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
		for stage_id in range(1, self.time_horizon+1, self.timestep_size):
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
			raise ValueError("Objective function must be provided before running DP. Either provide it as an argument"
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

		self.next_stage = next_stage
		self.previous_stage = previous
		self.matrix = None  # this will be created from the parameters when .optimize is run
		self.number = None

		self.pass_data = []
		self.choices_index = []

	def create_prior_handler(self):
		if self.prior_handler_class is not None:
			self.prior_handler = self.prior_handler_class(stage=self)

	def build(self):
		"""
			Builds the stage table - evaluates the objective function for each location, passing the various
			values to it so it can provide a result. Does *not* handle prior data though - that is handled when
			we actually run the DP. We separate this out so that people who want to intervene between generating
			the stage table and handling the prior can do so.
		:return:
		"""

		states = self.parent_dp._all_states
		state_kwargs = self.parent_dp._state_keys
		decisions = self.decision_variable.options
		self.matrix = numpy.zeros((len(states), len(decisions)))

		for row_id, state_values in enumerate(states):
			for column_id, decision_value in enumerate(decisions):
				state_data = dict(zip(state_kwargs, state_values))  # build the kwarg pairs into a dictionary we can pass into the objective function
				decision_data = {self.decision_variable.variable_id: decision_value}
				self.matrix[row_id][column_id] = self.parent_dp.objective_function(stage=self,
																					**support.merge_dicts(state_data,
																										  decision_data))

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name

	def optimize(self, prior=None):
		"""
			This handles the actual backward DP calculation - assumes that the stage has a matrix that is built
			already with appropriate values.

			This method will handle migrating prior data back through. As notes:
				* We should be able to do a lot of this with numpy broadcasting. If we take the calculation function
						of each row in the previous stage, then broadcast it across this stage, that should do it.
				* Need to think about where reducers fit in - if we have a probabilistic reducer, that can run after
						we do the broadcasting, I believe.
				* Need to figure out how we want to handle both selection constraints, but also decision/state interactions
						Right now, we can say which state a decision feeds back on - but what's going through my head about that
						is: 1) Do we need that if we expect an objective function? I think we do because that determines
							how we take values from the future and propogate them backward. This is where maybe we
							might not actually be able to use numpy broadcasting? Or maybe we have to shift the array
							we're broadcasting to align based on the best decision - not sure - need to think of how
							we're going to handle that a bit more
							2) What about where there's some limit imposed by decision var / state var interaction. For
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
		if self.prior_handler.exists():
			#prior_shaped = prior.reshape(prior.size, 1)  # reshape to the correct dimensions to broadcast it down the matrix
			self.matrix = self.prior_handler.apply()

		self.pass_data = self.parent_dp.calculation_function(self.matrix, axis=1)  # sum the rows and find the max
		self.choices_index = self.parent_dp.index_calculation_function(self.matrix, axis=1)  # get the column of the min/max value

		if self.previous_stage:
			self.previous_stage.optimize(self.pass_data)  # now run the prior stage

	def get_optimal_values(self, prior=0):
		"""
			After running the backward DP, moves forward and finds the best choices at each stage.
		:param prior: The value of the choice at each stage
		:return:
		"""

		if self.parent_dp.max_selections:  # this format probably only works for 1 state variable and will need to be re-engineered
			max_selections = self.parent_dp.max_selections
		else:
			max_selections = len(self.parent_dp._all_states)

		amount_remaining = max_selections - prior
		if amount_remaining > 0:
			available_options = self.pass_data[:amount_remaining]  # strip off the end of it to remove values that we can't use
			best_option = available_options[-1]  # get the last value
			row_of_best = numpy.where(available_options == best_option)  # now we need the actual row to use in the matrix

			if self.matrix.any():
				column_of_best = numpy.where(self.matrix[row_of_best[0]] == best_option)[-1].flatten()[0]  # get the column of the best option - also the number of days
			else:  # this triggers for the last item, which doesn't have a matrix, but just a costs list
				column_of_best = row_of_best + 1
		else:
			best_option = 0
			column_of_best = 0

		#if self.selection_constraints:
		#	number_of_items = max([0, column_of_best - self.selection_constraints[self.number]])  # take the max with 0 - if it's negative, it should be 0
		#else:
		self.decision_amount = self.parent_dp.decision_variable.options[column_of_best]
		log.info("{} - Decision Amount at Stage: {}, Total Cost/Benefit: {}".format(self.name, self.decision_amount, best_option))

		if self.next_stage:
			self.next_stage.get_optimal_values(prior=self.decision_amount + prior)


class Prior(object):
	"""
		A class for applying future stage values back to the previous stage
	"""
	def __init__(self, stage, data=None, matrix=None):
		self.data = data
		self.matrix = matrix
		self.parent_stage = stage

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
				#if column_value == 0 and row_index - column_index >= 0:  # we only need to calculate fields that meet this condition - the way it's calculated, others will cause an IndexError anyway
				stage_value = self.matrix[row_index][column_index]  # the value for this stage is on 1:1 line, so, it is where the indices are equal

				if stage_value == self.parent_stage.parent_dp.exclusion_value:
					continue

				try:
					prior_value = self.data[row_index-column_index]
					self.matrix[row_index+1][column_index] = stage_value + prior_value
				except IndexError:
					continue
		return self.matrix
