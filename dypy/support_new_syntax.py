# this is a separate file because if the code in it gets processed by an earlier version of python, it crashes,
# so we load it only when we have to

def _dict_merge_gte_35(dictionary1, dictionary2):
	return {**dictionary1, **dictionary2}