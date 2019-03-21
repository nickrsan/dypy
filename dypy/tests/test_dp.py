import unittest
import logging

import dypy
import logging_test  # configures logging

log = logging.getLogger(__name__)

class SimpleDPTest(unittest.TestCase):
	def setUp(self):

		decision_variable = dypy.DecisionVariable("Time on Course", variable_id="time_on_course", options=[1, 2, 3, 4])
		state_variable = dypy.StateVariable("Days Spent Studying", variable_id="days_spent", values=[1, 2, 3, 4, 5, 6, 7])

		benefit_list = [
			[0, 3, 5, 6, 7],
			[0, 5, 6, 7, 9],
			[0, 2, 4, 7, 8],
			[0, 6, 7, 9, 9],
		]

		dynamic_program = dypy.DynamicProgram(timestep_size=1, time_horizon=4, calculation_function=max)

		dynamic_program.decision_variable = decision_variable
		dynamic_program.add_state_variable(state_variable)

		dynamic_program.build_stages(name_prefix="Course")
		for index, stage in enumerate(dynamic_program.stages):
			stage.cost_benefit_list = benefit_list[index]  # set the benefit options for each stage

		dynamic_program.run()

	def test_basic(self):
		self.assertTrue(True)  # just want it to run setUp right now


class SimpleDPTestFunction(unittest.TestCase):
	"""
		Same as SimpleDPTest, but uses an objective function instead of modifying the stages on its own - we want
		both of these to work
	"""
	def test_course_studying(self):

		# we have 7 days available to us for studying
		state_variable = dypy.StateVariable("Days Spent Studying", values=[1, 2, 3, 4, 5, 6, 7])
		# but we can only spend between 1 and 4 days studying for any single course
		decision_variable = dypy.DecisionVariable("Time on Course", options=[1, 2, 3, 4])

		def objective_function(stage, days_spent_studying, time_on_course):
			"""
				When the objective is called, the solver makes the stage, state variables,
				and decision variable available to it.

				stage,
				state
				decision
			"""
			# we are given benefits, so defining here - each course is a row, each column is
			# number of days, and the value is how much benefit we get from studying that
			# course for that long.
			benefit_list = [
				[0, 3, 5, 6, 7],
				[0, 5, 6, 7, 9],
				[0, 2, 4, 7, 8],
				[0, 6, 7, 9, 9],
			]

			if time_on_course > days_spent_studying:
				return stage.parent_dp.exclusion_value
			else:
				return benefit_list[stage.number][time_on_course]

		# define the dynamic program and tell it we have four timesteps
		dynamic_program = dypy.DynamicProgram(timestep_size=1, time_horizon=4,
											  objective_function=objective_function, calculation_function=dypy.MAXIMIZE,
											  prior=dypy.SimplePrior)
		dynamic_program.exclusion_value = -1  # set it to -1 since we know nothing else will be negative here - let's us visualize arrays better
		dynamic_program.decision_variable = decision_variable
		dynamic_program.add_state_variable(state_variable)

		# each course will be a stage, in effect - tell it to create all four stages
		dynamic_program.build_stages(name_prefix="Course")
		dynamic_program.run()

		self.assertEqual(2, dynamic_program.stages[0].decision_amount)
		self.assertEqual(1, dynamic_program.stages[1].decision_amount)
		self.assertEqual(3, dynamic_program.stages[2].decision_amount)
		self.assertEqual(1, dynamic_program.stages[3].decision_amount)

	def test_programming_effort(self):
		"""
			A slightly larger problem that we use in the examples in the documentation - the results diverge from my simple
			solver code, so figuring out which is correct and building that into the test here will be important
		:return:
		"""
		# we have 12 days available to us for studying
		state_variable = dypy.StateVariable("Days Spent On Category", values=range(1, 13))
		# but we can only spend between 1 and 4 days studying for any single course
		decision_variable = dypy.DecisionVariable("Time on Category", options=range(1, 6))

		def objective_function(stage, days_spent_on_category, time_on_category):
			"""
				When the objective is called, the solver makes the stage, state variables,
				and decision variable available to it. The keyword argument names here
				match the names of the state variables and decision variables (which will be
				passed as keyword arguments, so order isn't important, but name is). The name
				is automatically derived by lowercasing the name of the variable and replacing
				spaces with underscores. It will remove digits from the front if they are present
				so that the name is a valid python identifier. You can also manually provide a
				kwarg name - if you wish to do this, see the API documentation.

				In this example, we can think of these variables as providing:
					1. stage - this will give access to the DyPy.stage object
					2. state - this just provides access to the state value being assessed, not the object
					3. decision - this just provides access to the decision value being assessed, not the object
			"""

			# we are given benefits, so defining here - each category is a row, each column is
			# number of days, and the value is how much benefit we get from working on that category
			# for that long
			benefit_list = [
				# column 0 means no time spent on it, so we get no benefit
				[0, 2, 5, 7, 8, 10],
				[0, 3, 6, 8, 9, 9],
				[0, 1, 2, 4, 7, 9],
				[0, 2, 3, 6, 8, 11],
				[0, 4, 6, 8, 9, 10],
			]

			if time_on_category > days_spent_on_category:
				return stage.parent_dp.exclusion_value  #
			else:
				return benefit_list[stage.number][time_on_category]

		# define the dynamic program and tell it we have four timesteps
		dynamic_program = dypy.DynamicProgram(timestep_size=1, time_horizon=5,
											  objective_function=objective_function, calculation_function=dypy.MAXIMIZE,
											  prior=dypy.SimplePrior, max_selections=12)
		dynamic_program.exclusion_value = -1  # set it to -1 since we know nothing else will be negative here - lets us visualize arrays better
		dynamic_program.decision_variable = decision_variable
		dynamic_program.add_state_variable(state_variable)

		# each category will be a stage, in effect - tell it to create all five stages as empty
		dynamic_program.build_stages(name_prefix="Category")  # assigns names by default, but we can override them
		stage_names = ["Core", "UI/UX", "Network", "Database", "API"]
		for i, stage in enumerate(dynamic_program.stages):
			dynamic_program.stages[i].name = stage_names[i]  # these need to match the order

		dynamic_program.run()
