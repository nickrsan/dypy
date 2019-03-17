from itertools import chain

import sys

def present_value(value, year, discount_rate, compounding_rate=1):
	"""
		Calculates the present value of a future value similar to numpy.pv, except numpy.pv gave weird negative values
	:param value: The future value to get the present value of
	:param year: How many years away is it?
	:param discount_rate: The discount rate to use for all years/periods in between
	:param compounding_rate: How often during the period should values be compounded. 1 means once a year, 365 means daily, etc.
	:return:  present value of the provided value
	"""
	return value * (1 + float(discount_rate)/float(compounding_rate)) ** (-year*compounding_rate)



def _dict_merge_lt_35(dictionary1, dictionary2):
	"""
		See https://treyhunner.com/2016/02/how-to-merge-dictionaries-in-python/
	:return:
	"""
	return dict(chain(dictionary1.items(), dictionary2.items()))


## Determine the best way to merge dicts - we do it this way for speed because this will be called for every cell in every stage  - the new method is significantly faster than the old
if sys.version_info.major == 3 and sys.version_info.minor >= 5:
	import support_new_syntax  # this is here because the code in this file is only readable by Python 3.5 and up
	merge_dicts = support_new_syntax._dict_merge_gte_35
else:
	merge_dicts = _dict_merge_lt_35