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

	in_files, out_file, algorithm = reels.handle_args()
	free, context = reels.setup(in_files)
	search = getattr(reels, algorithm)

	# Build root node
	# initialize leaf list with the root node
	# choose any obs as starting point for the solution
	cost = len(context.obs[free[0]]) - context.overmat[0][0]
	root = reels.ReelNode([free[0]], free[1:], cost)

	def print_goal_file(goal):
		solution_str = reels.final_solution(goal.sequence, context)
		out_fd.write(solution_str + '\n')

	def print_goal_stdout(goal):
		solution_str = reels.final_solution(goal.sequence, context)
		sys.stdout.write(solution_str + '\n')

	def _run(search, root, context, goal_callback):
		pro_file = 'out.profile'
		cProfile.runctx('search(root, context, goal_callback)', globals(), locals(), pro_file)
		p = pstats.Stats(pro_file)
		p.strip_dirs().sort_stats('filename','line',).print_stats()

	if out_file:
		out_fd = io.open(out_file, 'a')
		_run(search, root, context, print_goal_file)
		out_fd.close()
	else:
		_run(search, root, context, print_goal_stdout)

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

	in_files, out_file, algorithm = reels.handle_args()
	search = getattr(reels, algorithm)

	measurements = []

	def mute_goal(goal):
		pass # do not print anything

	for i in range(N_RUNS):
		if print_stuff: sys.stdout.write('.')
		free, context = reels.setup(in_files)

		# Build root node
		# initialize leaf list with the root node
		# choose any obs as starting point for the solution
		cost = len(context.obs[free[0]]) - context.overmat[0][0]
		root = reels.ReelNode([free[0]], free[1:], cost)

		t0 = time.time()
		result = search(root, context, mute_goal)
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
