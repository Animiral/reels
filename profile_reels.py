#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-
#!/usr/bin/env jython
#!/usr/bin/env python

'''Runs the cProfile profiler on reels.py.'''

import sys
import reels
import genreel
import pstats
import functools

def profile_reels():
	import cProfile

	args = reels.handle_args()
	read_obs_func = functools.partial(reels.read_obs_csv, dialect=args.dialect) if args.csv else reels.read_obs
	search = {'astar': reels.astar, 'dfs': reels.dfs} [args.algorithm]
	free, context = reels.setup(args.in_file, read_obs_func)
	out_fd = io.open(args.out_file, 'a') if args.out_file else sys.stdout
	format_solution = (lambda s: ','.join(s)) if args.csv else (lambda s: ''.join(s))

	# Build root node
	# initialize leaf list with the root node
	# choose any obs as starting point for the solution
	cost = len(context.obs[free[0]]) - context.overmat[0][0]
	root = reels.ReelNode([free[0]], free[1:], cost, context)

	def print_goal(goal):
		'''Count the number of calls to print_goal.
		If the search should continue, return True.
		If the print_count limit is exhausted, return False.
		'''
		solution = goal.final_solution(context)
		goal_str = format_solution(solution)
		out_fd.write(goal_str + '\n')
		print_goal.print_count = print_goal.print_count - 1
		return print_goal.print_count > 0

	print_goal.print_count = args.solutions

	# with out_fd: # close file when finished, come what may (exceptions etc)
	pro_file = 'out.profile'
	cProfile.runctx('search(root, context, print_goal, args.limit, args.full)', globals(), locals(), pro_file)
	p = pstats.Stats(pro_file)
	p.strip_dirs().sort_stats('filename','line',).print_stats()
	# out_fd.close()

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
	measurements = []

	args = reels.handle_args()
	read_obs_func = functools.partial(reels.read_obs_csv, dialect=args.dialect) if args.csv else reels.read_obs
	search = {'astar': reels.astar, 'dfs': reels.dfs} [args.algorithm]
	free, context = reels.setup(args.in_file, read_obs_func)

	def mute_goal(goal):
		pass # do not print anything

	for i in range(N_RUNS):
		if print_stuff: sys.stdout.write('.')
		free, context = reels.setup(args.in_file, read_obs_func)

		# Build root node
		# initialize leaf list with the root node
		# choose any obs as starting point for the solution
		cost = len(context.obs[free[0]]) - context.overmat[0][0]
		root = reels.ReelNode([free[0]], free[1:], cost, context)

		t0 = time.time()

		try:
			result = search(root, context, mute_goal, args.limit, args.full)
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
