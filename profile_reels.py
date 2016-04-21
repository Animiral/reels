#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-
#!/usr/bin/env jython
#!/usr/bin/env python

# This script runs the cProfile profiler on reels.py.

import sys
import reels
import genreel
import pstats

def reels_astar_wrapper():
	reels_astar_wrapper.result = reels.astar(reels_astar_wrapper.free, reels_astar_wrapper.context)

def profile_reels():
	import cProfile

	in_files, out_file = reels.handle_args()
	reels_astar_wrapper.free, reels_astar_wrapper.context = reels.setup(in_files)

	pro_file = 'out.profile'
	cProfile.run('reels_astar_wrapper()', pro_file)
	result = reels_astar_wrapper.result 

	if out_file:
		file = io.open(out_file, 'a')
		file.write(result + '\n')
		file.close()
	else:
		sys.stdout.write(result + '\n')

	p = pstats.Stats(pro_file)
	p.strip_dirs().sort_stats('filename','line',).print_stats()

def median(a):
	N = len(a)
	if (N % 2) == 0:
		return (a[N//2] + a[N//2+1]) / 2
	else:
		return a[N/2]

# returns median time
def time_reels(print_stuff):
	import time

	N_RUNS = 20

	in_files, out_file = reels.handle_args()

	measurements = []

	for i in range(N_RUNS):
		if print_stuff: sys.stdout.write('.')
		free, context = reels.setup(in_files)
		t0 = time.time()
		result = reels.astar(free, context)
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
