#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-
#!/usr/bin/env jython
#!/usr/bin/env python

# This script runs the cProfile profiler on reels.py.

import sys
import reels
import genreel
import pstats

def random_profile():
	import cProfile
	#       reel_size, sym_count, obs_count, obs_min, obs_max
	args = (   50    ,    10    ,    10    ,    5   ,    8   )
	reel, obs = genreel.random_observations(*args)
	obs_list = list(obs)
	sys.stdout.write('Start profiling with reel="{0}", obs_list={1}...\n'.format(reel, obs_list))

	reels.g_obs = obs_list
	reels.setup()
	cProfile.run(__name__ + '.result = reels.astar()')

	sys.stdout.write('Done.\n')
	sys.stdout.write('result="{0}"\n'.format(result))

def reels_astar_wrapper():
	global result
	result = reels.astar()

def profile_reels():
	import cProfile
	global result

	reels.handle_args()

	if reels.g_run_tests:
		raise RuntimeException('Tests option disabled for profiler.')

	reels.g_obs = reels.read_obs(reels.g_files)
	reels.setup()

	pro_file = 'out.profile'
	result = ''
	# cProfile.runctx('result = reels.astar()', reels.__dict__, __dict__, pro_file)
	cProfile.run('reels_astar_wrapper()', pro_file)

	if reels.g_out_file:
		out_file = io.open(reels.g_out_file, 'a')
		out_file.write(result + '\n')
		out_file.close()
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
	# import statistics
	import time

	N_RUNS = 20

	reels.handle_args()
	if reels.g_run_tests:
		raise RuntimeException('Tests option disabled for timing.')
	reels.g_obs = reels.read_obs(reels.g_files)

	measurements = []

	for i in range(N_RUNS):
		if print_stuff: sys.stdout.write('.')
		reels.setup()
		t0 = time.time()
		reels.astar()
		t1 = time.time()
		measurements.append(t1-t0)

	# median_time = statistics.median(measurements)
	median_time = median(measurements)
	if print_stuff: sys.stdout.write('measurements={0}, median time = '.format(measurements))
	sys.stdout.write('{0:.3f}\n'.format(median_time))

def main():
	import sys

	if sys.argv[1] == 'p': # run profile
		sys.argv = sys.argv[:1] + sys.argv[2:]
		profile_reels()

	if sys.argv[1] == 't': # run timer
		sys.argv = sys.argv[:1] + sys.argv[2:]
		time_reels(print_stuff=True)

	if sys.argv[1] == 'ts': # run timer, succinct/simple output (time only)
		sys.argv = sys.argv[:1] + sys.argv[2:]
		time_reels(print_stuff=False)



if __name__ == "__main__":
	main()

## OUTPUTS ##

# [peter@ad-pc reels]$ pypy3 ./profile_reels.py p in/problem25.in 
# 5723a14769961b87a26a834aa47aab04b01857b441
# Tue Apr 12 09:38:33 2016    out.profile

#          42832073 function calls in 120.372 seconds

#    Ordered by: file name, line number

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    0.000    0.000  120.371  120.371 <string>:1(<module>)
#     28063    0.004    0.000    0.010    0.000 __init__.py:1230(debug)
#         2    0.000    0.000    0.000    0.000 __init__.py:1242(info)
#     28065    0.003    0.000    0.003    0.000 __init__.py:1470(getEffectiveLevel)
#     28065    0.003    0.000    0.006    0.000 __init__.py:1484(isEnabledFor)
#         2    0.000    0.000    0.000    0.000 __init__.py:1770(info)
#     28063    0.012    0.000    0.023    0.000 __init__.py:1780(debug)
#    264566    1.012    0.000  109.435    0.000 bisect.py:3(insort_right)
#         1    0.000    0.000  120.371  120.371 profile_reels.py:25(reels_astar_wrapper)
#    341924    0.664    0.000    0.664    0.000 reels.py:50(solution)
#         3    0.000    0.000    0.000    0.000 reels.py:68(final_solution)
#    313858    0.015    0.000    0.015    0.000 reels.py:90(__init__)
#   4359464    0.974    0.000    1.212    0.000 reels.py:100(__lt__)
#     70519    0.179    0.000    0.179    0.000 reels.py:108(__eq__)
#    892282    0.692    0.000    0.692    0.000 reels.py:115(__hash__)
#    313856    1.983    0.000    5.048    0.000 reels.py:127(est)
#   8148388    0.240    0.000    0.240    0.000 reels.py:146(<lambda>)
#   8148388    0.257    0.000    0.257    0.000 reels.py:147(<lambda>)
#    341920    0.359    0.000    5.840    0.000 reels.py:163(successor)
#    313858    0.406    0.000    5.480    0.000 reels.py:168(_make_successor)
#    313858    1.046    0.000    2.817    0.000 reels.py:195(purify)
#   1545412    1.192    0.000    1.192    0.000 reels.py:199(<lambda>)
#    313858    0.568    0.000  110.873    0.000 reels.py:229(connect)
#     28063    0.004    0.000    0.008    0.000 reels.py:243(pop)
#         1    0.723    0.723  120.371  120.371 reels.py:343(astar)
#    264566  107.203    0.000  107.203    0.000 {method 'insert' of 'list' objects}
#   7727060    2.515    0.000    3.013    0.000 {built-in function max}
#         1    0.000    0.000  120.372  120.372 {built-in function exec}
#     28063    0.005    0.000    0.005    0.000 {method 'pop' of 'list' objects}
#   8989902    0.311    0.000    0.311    0.000 {built-in function len}
#         1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

# [peter@ad-pc reels]$ pypy3 profile_reels.py p in/problem25.in 
# 6a834aa474415723a14769961baab04b01857b87a2
# Tue Apr 12 14:15:58 2016    out.profile

#          44968675 function calls in 13.639 seconds

#    Ordered by: file name, line number

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    0.000    0.000   13.638   13.638 <string>:1(<module>)
#     41190    0.004    0.000    0.013    0.000 __init__.py:1230(debug)
#         2    0.000    0.000    0.000    0.000 __init__.py:1242(info)
#     41192    0.004    0.000    0.004    0.000 __init__.py:1470(getEffectiveLevel)
#     41192    0.005    0.000    0.009    0.000 __init__.py:1484(isEnabledFor)
#         2    0.000    0.000    0.000    0.000 __init__.py:1770(info)
#     41190    0.013    0.000    0.027    0.000 __init__.py:1780(debug)
#    373794    0.057    0.000    0.671    0.000 heapq.py:133(heappush)
#     41190    0.012    0.000    1.094    0.000 heapq.py:138(heappop)
#    414983    0.481    0.000    0.588    0.000 heapq.py:236(_siftdown)
#     41189    0.583    0.000    1.079    0.000 heapq.py:289(_siftup)
#         1    0.000    0.000   13.638   13.638 profile_reels.py:28(reels_astar_wrapper)
#    486037    0.655    0.000    0.655    0.000 reels.py:51(solution)
#       101    0.000    0.000    0.001    0.000 reels.py:69(final_solution)
#    444746    0.020    0.000    0.020    0.000 reels.py:91(__init__)
#    281455    0.235    0.000    0.251    0.000 reels.py:102(__lt__)
#    384796    0.509    0.000    0.509    0.000 reels.py:110(__eq__)
#   1263286    0.745    0.000    0.745    0.000 reels.py:117(__hash__)
#    444646    2.045    0.000    5.698    0.000 reels.py:129(est)
#  10817784    0.343    0.000    0.343    0.000 reels.py:148(<lambda>)
#  10817784    0.349    0.000    0.349    0.000 reels.py:149(<lambda>)
#    485935    0.271    0.000    6.379    0.000 reels.py:165(successor)
#    444746    0.371    0.000    6.106    0.000 reels.py:170(_make_successor)
#    444746    1.214    0.000    3.179    0.000 reels.py:197(purify)
#   2113680    1.376    0.000    1.376    0.000 reels.py:201(<lambda>)
#    444746    0.618    0.000    2.241    0.000 reels.py:231(connect)
#     41190    0.008    0.000    1.102    0.000 reels.py:246(pop)
#         1    0.643    0.643   13.638   13.638 reels.py:347(astar)
#  10568400    2.894    0.000    3.586    0.000 {built-in function max}
#         1    0.000    0.000   13.638   13.638 {built-in function exec}
#    373794    0.050    0.000    0.050    0.000 {method 'append' of 'list' objects}
#   4033684    0.130    0.000    0.130    0.000 {built-in function len}
#     41190    0.002    0.000    0.002    0.000 {method 'pop' of 'list' objects}
#         1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}





# ....................measurements=[7.767170190811157, 6.684993028640747, 6.467377185821533, 6.91469407081604, 6.240242958068848,
# 6.742031812667847, 6.670589923858643, 6.693011999130249, 6.530810117721558, 6.583081960678101, 6.975265979766846, 6.5542120933532715,
# 6.139000177383423, 6.860619068145752, 6.340721130371094, 6.714582204818726, 6.734888792037964, 6.682585000991821, 6.724730968475342, 6.631833076477051], 
# median time = 6.765
