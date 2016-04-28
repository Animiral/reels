#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-
#!/usr/bin/env jython
#!/usr/bin/env python

'''Runs the cProfile profiler on reels.py.'''

import sys
import reels
import genreel
import pstats

def profile_reels():
	import cProfile

	in_file, out_file, algorithm, make_obs_func, solutions, limit, full = reels.handle_args()
	search = getattr(reels, algorithm)
	free, context = reels.setup(in_file, make_obs_func)

	# Build root node
	# initialize leaf list with the root node
	# choose any obs as starting point for the solution
	cost = len(context.obs[free[0]]) - context.overmat[0][0]
	root = reels.ReelNode([free[0]], free[1:], cost, context)

	def abort_after_n(print_func):
		'''Decorate the print_func with a countdown to raise AbortSearch after the limit is reached.'''

		def wrapped(goal):
			print_func(goal)
			wrapped.n = wrapped.n - 1
			if wrapped.n <= 0:
				raise reels.AbortSearch()

		wrapped.n = solutions

		return wrapped

	@abort_after_n
	def print_goal_file(goal):
		'''Print the solution from the goal node to the open file out_fd.'''
		solution_str = goal.final_solution(context)
		out_fd.write(solution_str + '\n')

	@abort_after_n
	def print_goal_stdout(goal):
		'''Print the solution from the goal node to stdout.'''
		solution_str = goal.final_solution(context)
		sys.stdout.write(solution_str + '\n')

	def _run(search, root, context, goal_callback, limit, full):
		pro_file = 'out.profile'
		cProfile.runctx('search(root, context, goal_callback)', globals(), locals(), pro_file)
		p = pstats.Stats(pro_file)
		p.strip_dirs().sort_stats('filename','line',).print_stats()

	try:
		if out_file:
			with io.open(out_file, 'a') as out_fd:
				_run(search, root, context, print_goal_file, limit, full)
		else:
			search(root, context, print_goal_stdout, limit, full)
	except reels.AbortSearch:
		pass # successfully aborted search

def median(a):
	N = len(a)
	if (N % 2) == 0:
		return (a[N//2] + a[N//2+1]) / 2
	else:
		return a[N/2]

def time_reels(print_stuff):
	'''Returns median time.'''
	import time

	N_RUNS = 20

	in_file, out_file, algorithm, make_obs_func, solutions, limit, full = reels.handle_args()
	search = getattr(reels, algorithm)

	measurements = []

	def mute_goal(goal):
		pass # do not print anything

	for i in range(N_RUNS):
		if print_stuff: sys.stdout.write('.')
		free, context = reels.setup(in_file, make_obs_func)

		# Build root node
		# initialize leaf list with the root node
		# choose any obs as starting point for the solution
		cost = len(context.obs[free[0]]) - context.overmat[0][0]
		root = reels.ReelNode([free[0]], free[1:], cost, context)

		t0 = time.time()

		try:
			result = search(root, context, mute_goal)
		except reels.AbortSearch:
			pass # successfully aborted search

		t1 = time.time()
		measurements.append(t1-t0)

	median_time = median(measurements)
	if print_stuff: sys.stdout.write('measurements={0}, median time = '.format(measurements))
	sys.stdout.write('{0:.3f}\n'.format(median_time))

def main():
	import sys

	mode = sys.argv[1]
	sys.argv = sys.argv[:1] + sys.argv[2:] # strip mode from args so that reels.py can handle them

	if mode == 'p': # run profile
		profile_reels()

	elif mode == 't': # run timer
		time_reels(print_stuff=True)

	elif mode == 'ts': # run timer, succinct/simple output (time only)
		time_reels(print_stuff=False)

	else:
		sys.stderr.write('"{0}" is not a valid mode.\n'.format(mode))

if __name__ == "__main__":
	main()
