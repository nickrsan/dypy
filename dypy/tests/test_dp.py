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
	def setUp(self):

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
		self.dynamic_program = dypy.DynamicProgram(timestep_size=1, time_horizon=4,
											  objective_function=objective_function, calculation_function=dypy.MAXIMIZE,
											  prior=dypy.SimplePrior)
		self.dynamic_program.exclusion_value = -1  # set it to -1 since we know nothing else will be negative here - let's us visualize arrays better
		self.dynamic_program.decision_variable = decision_variable
		self.dynamic_program.add_state_variable(state_variable)

		# each course will be a stage, in effect - tell it to create all four stages
		self.dynamic_program.build_stages(name_prefix="Course")
		self.dynamic_program.run()

	def test_basic(self):
		self.assertEqual(2, self.dynamic_program.stages[0].decision_amount)
		self.assertEqual(1, self.dynamic_program.stages[1].decision_amount)
		self.assertEqual(3, self.dynamic_program.stages[2].decision_amount)
		self.assertEqual(1, self.dynamic_program.stages[3].decision_amount)
