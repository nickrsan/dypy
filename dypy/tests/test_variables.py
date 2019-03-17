import unittest

import dypy


class VariableSupportTest(unittest.TestCase):
	def test_check_variable_id(self):
		"""
			Makes sure that when no variable_id is provided, we get the names
			we expect
		:return:
		"""

		name_id_pairs = {
			"variable_id": "variable_id",
			"Variable_id": "variable_id",
			"Variable ID": "variable_id",
			"5three Variable": "three_variable",
			"Remove-my-hyphens": "remove_my_hyphens",
		}

		leave_capitalization_intact = {
			"variable_id": "variable_id",
			"Variable_id": "Variable_id",
			"Variable ID": "Variable_ID",
			"5three Variable": "three_Variable",
			"Remove-my-hyphens": "Remove_my_hyphens",
		}

		for key in name_id_pairs:
			self.assertEqual(name_id_pairs[key], dypy.variables.check_variable_id(key, None))

		for key in leave_capitalization_intact:
			self.assertEqual(leave_capitalization_intact[key], dypy.variables.check_variable_id(key, key))
