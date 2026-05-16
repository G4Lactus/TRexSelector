import trex_selector as ts
ts.core.utils.set_num_threads(2)
print("Should see max threads info:")
ts.core.utils.get_max_threads() # wait, this doesn't log.
