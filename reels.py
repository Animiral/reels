#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

# The file README contains a general description of the program. Read on for implementation details.

# reels.py is an A* search algorithm.

# Nodes in the search graph consist of a partial solution built of one or more observed pieces and a list of free pieces, which are not yet represented in the solution.
# In the start node, the partial solution contains only the largest single observed piece of the reel. All other pieces are free.
# In the goal nodes, the solution is a complete sequence of observed pieces, some possibly overlapping with the next. The free list is empty.

# To move from one node to the next, the program appends a free piece to the left or right of the partial solution.
# The cost of the edge is the number of additional symbols in the partial solution after the operation.
# Overlapping observations are therefore preferred to find a short solution.

# To help with determining the cheapest path, the program uses an NxN matrix g_overlap, where N is the number of observed pieces.
# g_overlap[i][j] is the number of overlapping symbols when appending g_obs[j] to the right of g_obs[i].

# The heuristic function (estimated distance to goal) is the sum of minimum costs using g_overlap_left and g_overlap_right for every free piece.

import sys
import io
import argparse
import fileinput
import re
import logging
import heapq
import itertools
import collections

# Context holds information about the search environment.
# obs is the list of observation strings taken from the source reel.
# overmat is the overlap matrix of precomputed overlaps between observations.
Context = collections.namedtuple('Context', ['obs', 'overmat'])

# NOTE: Lists of observations are generally implemented as lists of indexes into the g_obs list.

# Returns the number of overlapping symbols when appending the right piece to the left piece
def overlap(left, right):
	max_overlap = min(len(left),len(right)) - 1 # legal pieces can not overlap whole

	for i in range(max_overlap, 0, -1):
		if left[-i:] == right[:i]:
			return i
	else:
		return 0

# Returns the solution string from a list of obs indices representation
def solution(sequence, context):
	obs, overmat = context

	prev_index = sequence[0]
	S = obs[prev_index]

	for next_index in sequence[1:]:
		next_piece = obs[next_index]
		savings = overmat[prev_index][next_index]
		S += next_piece[savings:]
		prev_index = next_index

	return S

# Returns the final solution string from a list of obs indices representation.
# The difference to an intermediate solution is that due to its looped nature,
# overlap between the first and last piece can cut off some characters in the final solution.
def final_solution(sequence, context):
	S = solution(sequence, context)

	loop_overlap = context.overmat[sequence[-1]][sequence[0]] # Careful: don’t repeat the start of the solution if it overlaps with the end piece
	if loop_overlap > 0:
		return S[:-loop_overlap]
	else:
		return S 

# The nodes in the search graph are partially solved reel problems.
# A node is impure if any one free piece is a proper substring of the partial solution.
# In that case, it should be purified before searching on.
# For two nodes n1 and n2, (not n1 < n2) and (not n1 > n2) is not equivalent to n1 == n2.
# This is deliberate because node equality and ordering are used for different, unrelated purposes.
class ReelNode:
	sequence = [] # list of the observations included in the solution (in order)
	free = []     # set of free observations (also a list)
	cost = 0      # path cost of the partial solution that is the sequence = number of symbols in solution(sequence).
	est = 0       # (under-)estimated number of symbols in the solution reel, if we explore further expansions from this node

	def __init__(self,s,f,c):
		self.sequence = s
		self.free = f
		self.cost = c

	# String view for debugging: the cost and est is not important; we want to know the partial solution and free pieces
	def __str__(self):
		return 'ReelNode({0},{1})'.format(self.sequence, self.free)

	# Ordering for leaf heap.
	def __lt__(self, other):
		if self.est == other.est:
			return len(self.sequence) > len(other.sequence) # if tied, prefer to examine more complete solutions first
		else:
			return self.est < other.est

	# Equality: used for the node hash table in ReelGraph.
	# Nodes may be re-discovered with lower est, but they must still go in the same hash bucket.
	def __eq__(self, other):
		return self.sequence == other.sequence and self.free == other.free

	# Node hashing: XOR the bits of every item in the partial solution and the free list.
	# The bits are rotated around the integer after each XOR to make better use of the available space in the higher-order bits.
	# Partial solution and free list use different rotation phases so that moving the
	# first free piece to the end of the partial solution will produce a different hash.
	def __hash__(self):
		h = 0

		for s in self.sequence:
			h = ((h << 4) ^ (h >> 28)) ^ s

		for f in self.free:
			h = ((h << 5) ^ (h >> 27)) ^ s

		return h

# Returns the est of a node f(n) = g(n) + h(n), where g is the cost so far and h is the estimated cost to goal.
# g(n) is the number of symbols in the partial solution which are certainly a unique part of the final solution.
# Some possible overlapping symbols between the last and first piece of the partial solution are not counted
# towards the cost so far because even with some free pieces left over, the final reel might already be complete.
# h(n) is the estimated number of symbols that the free pieces are going to contribute towards the final reel.
# As an optimistic guess, we assume that every free piece will be placed such that it achieves optimal overlap
# with another piece to the left. The other piece could be any of the other free pieces or the tail of the sequence.
# Additionally, we discount a number of symbols according to the best possible overlap to the head of the sequence.
def est(node, context):
	obs, overmat = context

	G = node.cost

	free = node.free
	if not free: return G   # this is goal node

	sequence = node.sequence
	A = sequence[0]
	Z = sequence[-1]

	H = overmat[Z][A]                            # revert finished-loop assumption from cost g(n)
	H -= max(map(lambda a: overmat[a][A], free)) # assume best overlap from any free piece to A

	for f in free:
		free_Z = itertools.chain(free, [Z])
		max_overlap = max(map(lambda a: overmat[a][f], free_Z))
		H += len(obs[f]) - max_overlap

	H = max(0,H)

	return G + H

# Generates all successors to the given node
def successor(node, context):
	obs, overmat = context

	overlap = 0

	for i in range(0, len(node.free)):
		P = node.free[i]       # new piece
		A = node.sequence[0]   # sequence first piece
		Z = node.sequence[-1]  # sequence last piece

		# append free piece after partial solution
		sequence = node.sequence + [P]
		free = node.free[:i] + node.free[i+1:]
		cost = node.cost + len(obs[P]) + overmat[Z][A] - overmat[Z][P] - overmat[P][A]

		succ = ReelNode(sequence, free, cost)
		if free: succ = purify(succ, context)
		succ.est = est(succ, context)

		yield succ

# From a candidate graph node, removes every free piece that is a proper substring of the partial solution.
# These pieces do not have to be considered any longer and may even introduce errors.
def purify(node, context):
	obs = context.obs
	sol = solution(node.sequence, context)
	not_redundant = lambda f: obs[f] not in sol
	node.free = list(filter(not_redundant, node.free))

	return node

# Parse and handle command arguments
def handle_args():
	import multiprocessing
	
	parser = argparse.ArgumentParser(description='''
Reads input from FILES in order and writes a one-line result for each input file to OUTPUT.
If no input files are specified, reads from standard input.
If no output files are specified, writes to standard output.
		''')
	parser.add_argument('files', metavar='FILE', nargs='*', help='input file(s)')
	parser.add_argument('-o', '--out_file', help='append solution to this file')
	parser.add_argument('-p', '--processes', type=int, default=multiprocessing.cpu_count(), help='number of parallel worker processes to use')
	parser.add_argument('-s', '--size', type=int, default=100, help='number of paths to produce as a unit of work in one worker process')
	parser.add_argument('-d', '--debug', dest='log_level', action='store_const', const=logging.DEBUG, default=logging.WARNING, help=argparse.SUPPRESS)

	args = parser.parse_args()

	logging.basicConfig(level=args.log_level, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%T')

	return args.files, args.out_file, args.processes, args.size

# Reads reel observations from the files in the parameter list.
# Returns the resulting list of observations.
def make_obs(*in_files):
	obs = []

	for line in fileinput.input(*in_files):
		# remove newline at the end
		line = line.strip()

		# ignore empty lines
		if not line:
			continue

		# lines are invalid if they contain non-word characters (invalid symbols)
		if not re.match('^\w+$', line):
			raise RuntimeError('Illegal symbol in "{0}"'.format(line))

		obs.append(line)

	if not obs: raise RuntimeError('No input was given!')
	
	# prepare obs: remove duplicates (avoids both getting discarded as redundant later)
	obs = list(set(obs))
	obs.sort() # DEBUG: establish deterministic order of obs

	return obs

def make_overmat(obs):
	N = len(obs)
	elim = set() # redundant pieces

	overmat = [[0] * N for i in range(0,N)] # overlap matrix

	for i in range(0,N):
		for j in range(0,N):
			obs_i = obs[i]
			obs_j = obs[j]

			# mark redundant piece if we find any
			if (i != j) and (obs_i in obs_j):
				elim.add(i)

			overmat[i][j] = overlap(obs_i, obs_j)

	return overmat, elim

# Prepares data structures for search: obs list, overlap matrix and free list for start node
def setup(in_files):
	logging.info('SETUP from %s...', list(in_files))

	obs = make_obs(in_files)
	overmat, elim = make_overmat(obs)
	free = [i for i in range(0, len(obs)) if i not in elim] # eliminate initial pieces with complete overlap to reduce search space
	context = Context(obs, overmat)

	# logging.debug('obs is now %s (eliminated %s).', list(map(lambda x: obs[x], free)), elim)
	# logging.debug('overmat = %s', overmat)

	logging.info('SETUP DONE')
	return free, context

# This function runs a parallelizable unit of work in the search algorithm.
# It searches from the root node until it either finds a goal or the local number
# of open paths (leaves) goes above the given size.
# The return value is a heap of all open leaves if no goal was found, or a
# one-element list containing only the cheapest goal.
# TODO: test those conditions (atm it just does one search step)
def astar_part(root, size):
	import os

	global g_context

	logger = logging.getLogger(str(os.getpid()))

	# obs, overmat = g_context
	logger.debug('START work on %s', root)

	leaf = [root]

	while 0 < len(leaf) < size:
		cursor = heapq.heappop(leaf)
		if not cursor.free:    # this is goal node
			return [cursor]

		for s in successor(cursor, g_context):
			heapq.heappush(leaf, s)

	logger.debug('DONE => %s leaves', len(leaf))

	return leaf

# This is the main search algorithm.
# It searches the problem space (tree) for the cheapest goal, starting from the root, and returns one such goal node.
# It uses N child processes as workers, where N is given in the workers parameter.
# One unit of work that is delegated to a child process is bounded approximately by the size given in the size parameter.
def astar(root, context, workers, size):
	import concurrent.futures

	global g_context

	logging.info('ASTAR...')

	g_context = context  # prepare global to be used by all child processes

	leaf = [root]                     # heap of open nodes to explore
	future = set()                    # searching processes
	goal = ReelNode([],[],0)
	goal.est = sum(map(len,context.obs))   # upper bound for goal nodes

	# n_leaf = 100 # DEBUG counter for leaves
	# min_est = root.est
	# max_est = root.est

	with concurrent.futures.ProcessPoolExecutor(workers) as executor:
		f = executor.submit(astar_part, heapq.heappop(leaf), workers) # in first iteration, generate just enough size=children to feed the workers
		future.add(f)     # start of search

		while(future):
			logging.debug('Waiting for %s astar_parts...', len(future))

			done, future = concurrent.futures.wait(future, return_when=concurrent.futures.FIRST_COMPLETED)

			logging.debug('... %s DONE', len(done))

			for f in done:
				leaf = heapq.merge(leaf, f.result())

			leaf = list(leaf)

			logging.debug('Now %s leaves', len(leaf))

			# The more new_tasks we handle at once, the better the parallelization.
			# However, we also risk that more work goes to waste because of exploring suboptimal paths.
			i = len(future)
			while i < workers and leaf:
				cursor = heapq.heappop(leaf)
				if cursor.est < goal.est:
					if cursor.free:
						f = executor.submit(astar_part, cursor, size)
						future.add(f)
						i = i + 1
					else:
						goal = cursor

			# 	min_est = min(s.est, min_est) # DEBUG tracking
			# 	max_est = max(s.est, max_est) # DEBUG tracking

			# if len(leaf) > n_leaf:           # DEBUG report mem usage
			# 	logging.debug('len(leaf) = %s (d: %s ~ %s)', len(leaf), min_est, max_est)
			# 	n_leaf = len(leaf) * 1.5

			# logging.debug('Examine d=%s\tf(n)=%s\t%s\t%s)', len(cursor.sequence), cursor.est, solution(cursor.sequence), cursor)

	logging.info('ASTAR DONE')
	return goal

# Program entry point
def main():
	in_files, out_file, workers, size = handle_args()
	free, context = setup(in_files)

	# initialize the root node
	cost = len(context.obs[free[0]]) - context.overmat[0][0]
	root = ReelNode([free[0]], free[1:], cost)
	root = purify(root, context)
	root.est = est(root, context) # NOTE: this is only useful for debug output because there is no other node to choose from at first

	goal = astar(root, context, workers, size)
	result = final_solution(goal.sequence, context)

	if out_file:
		out_file = io.open(out_file, 'a')
		out_file.write(result + '\n')
		out_file.close()
	else:
		sys.stdout.write(result + '\n')

if __name__ == "__main__":
	main()
