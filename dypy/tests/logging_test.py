import logging
import sys

root_log = logging.getLogger()
if len(root_log.handlers) == 0:  # if we haven't already set up logging, set it up
	root_log.setLevel(logging.DEBUG)
	log_stream_handler = logging.StreamHandler(sys.stdout)
	log_stream_handler.setLevel(logging.DEBUG)
	log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	log_stream_handler.setFormatter(log_formatter)
	root_log.addHandler(log_stream_handler)

log = logging.getLogger("dypy.tests")