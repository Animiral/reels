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
import logging

class AbortSearch(Exception):
	pass

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

	def beat(node_count=0):
		'''This function gets called by the search algorithm in regular intervals.
		It ensures that the search complies with the space and time resource limits.
		If either the processing time or memory are exhausted, immediately abort
		the program with exit code 1.

		If the search algorithm provides a count of its memorized nodes
		(A* keeps a heap of open nodes), it is checked against the memsize.
		'''
		if beat.timeout and time.time() > beat.cutoff_time:
			logging.error('Search exceeded the timeout of %s seconds.', beat.timeout)
			raise AbortSearch()

		if beat.memsize and node_count > beat.memsize:
			logging.error('Search exceeded the memory limit of %s nodes.', beat.memsize)
			raise AbortSearch()

	if args.timeout:
		cutoff_time = time.time() + args.timeout

	beat.timeout = args.timeout
	beat.memsize = args.memsize
	beat.cutoff_time = cutoff_time

	print_goal.print_count = args.solutions

	pro_file = 'out.profile'
	try:
		cProfile.runctx('search(root, context, print_goal, args.sym_limit, args.full)', globals(), locals(), pro_file)
	except AbortSearch:
		pass # search killed by resource exhaustion
	p = pstats.Stats(pro_file)
	p.strip_dirs().sort_stats('filename','line',).print_stats()

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

	def beat(node_count=0):
		'''This function gets called by the search algorithm in regular intervals.
		It ensures that the search complies with the space and time resource limits.
		If either the processing time or memory are exhausted, immediately abort
		the program with exit code 1.

		If the search algorithm provides a count of its memorized nodes
		(A* keeps a heap of open nodes), it is checked against the memsize.
		'''
		if beat.timeout and time.time() > beat.cutoff_time:
			logging.error('Search exceeded the timeout of %s seconds.', beat.timeout)
			raise AbortSearch()

		if beat.memsize and node_count > beat.memsize:
			logging.error('Search exceeded the memory limit of %s nodes.', beat.memsize)
			raise AbortSearch()

	beat.timeout = args.timeout
	beat.memsize = args.memsize

	for i in range(N_RUNS):
		if print_stuff: sys.stdout.write('.')
		free, context = reels.setup(args.in_file, read_obs_func)

		if args.timeout:
			cutoff_time = time.time() + args.timeout
			beat.cutoff_time = cutoff_time

		# Build root node
		# initialize leaf list with the root node
		# choose any obs as starting point for the solution
		cost = len(context.obs[free[0]]) - context.overmat[0][0]
		root = reels.ReelNode([free[0]], free[1:], cost, context)

		try:
			t0 = time.time()
			_,_,_ = search(root, context, mute_goal, args.sym_limit, args.full, beat)
			t1 = time.time()
			measurements.append(t1-t0)
		except AbortSearch:
			measurements.append(sys.maxsize) # if more than 50% of runs violate the cutoff_time, median will be maxsize

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
