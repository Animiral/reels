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

g_obs = []               # list of observation strings/pieces
g_overlap = []           # matrix with precomputed overlap results between pieces

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.WARNING)

# NOTE: Lists of observations are generally implemented as lists of indexes into the g_obs list.

# Returns the number of overlapping symbols when appending the right piece to the left piece
def overlap(left, right):
	max_overlap = min(len(left),len(right)) - 1 # legal pieces can not overlap whole

	for i in range(max_overlap, 0, -1):
		if left[-i:] == right[:i]:
			return i
	else:
		return 0

# Returns the solution string from a list of g_obs indices representation
def solution(sequence):
	global g_obs
	global g_overlap

	prev_index = sequence[0]
	S = g_obs[prev_index]

	for next_index in sequence[1:]:
		next_piece = g_obs[next_index]
		savings = g_overlap[prev_index][next_index]
		S += next_piece[savings:]
		prev_index = next_index

	return S

# Returns the final solution string from a list of g_obs indices representation.
# The difference to an intermediate solution is that due to its looped nature,
# overlap between the first and last piece can cut off some characters in the final solution.
def final_solution(sequence):
	global g_overlap

	S = solution(sequence)

	loop_overlap = g_overlap[sequence[-1]][sequence[0]] # Careful: don’t repeat the start of the solution if it overlaps with the end piece
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
def est(node):
	global g_obs
	global g_overlap

	# if node.free:
	# 	sol = solution(node.sequence)
	# else:
	# 	sol = final_solution(node.sequence)
	# G = len(sol)
	G = node.cost

	# compute heuristic distance to goal by assuming that we will get optimal overlap for every free piece
	H = 0
	# left_anchor = [node.sequence[0]] + node.free
	# right_anchor = [node.sequence[-1]] + node.free

	node_free = node.free
	sequence = node.sequence
	for f in node_free:
		left_max = max(g_overlap[f][sequence[0]], max(map(lambda a: g_overlap[f][a], node_free)))
		right_max = max(g_overlap[sequence[-1]][f], max(map(lambda a: g_overlap[a][f], node_free)))
		H += max(0, len(g_obs[f]) - left_max - right_max)

	# left_anchor = [node.sequence[0]] + node.free
	# right_anchor = [node.sequence[-1]] + node.free

	# for f in node.free:
	# 	left = lambda a: g_overlap[f][a]
	# 	right = lambda a: g_overlap[a][f]
	# 	left_max = max(map(left, left_anchor))
	# 	right_max = max(map(right, right_anchor))
	# 	H += max(0, len(g_obs[f]) - left_max - right_max)

	return G + H

# Generates all successors to the given node
def successor(node):
	global g_obs

	overlap = 0

	for i in range(0, len(node.free)):
		P = node.free[i]       # new piece
		A = node.sequence[0]   # sequence first piece
		Z = node.sequence[-1]  # sequence last piece

		# append free piece after partial solution
		sequence = node.sequence + [P]
		free = node.free[:i] + node.free[i+1:]
		cost = node.cost + len(g_obs[P]) + g_overlap[Z][A] - g_overlap[Z][P] - g_overlap[P][A]

		succ = ReelNode(sequence, free, cost)
		if free: succ = purify(succ)
		succ.est = est(succ)

		yield succ

# From a candidate graph node, removes every free piece that is a proper substring of the partial solution.
# These pieces do not have to be considered any longer and may even introduce errors.
def purify(node):
	global g_obs

	sol = solution(node.sequence)
	not_redundant = lambda f: g_obs[f] not in sol
	node.free = list(filter(not_redundant, node.free))

	return node

# g_graph = None          # search graph structure
g_files = []            # list of input files
g_run_tests = False     # whether to run unit tests
g_out_file = ''         # output file name

# Parse and handle command arguments
def handle_args():
	global g_files
	global g_run_tests
	global g_out_file

	parser = argparse.ArgumentParser(description='''
Reads input from FILES in order and writes a one-line result for each input file to OUTPUT.
If no input files are specified, reads from standard input.
If no output files are specified, writes to standard output.
		''')
	parser.add_argument('files', metavar='FILE', nargs='*', help='input file(s)')
	parser.add_argument('-o', '--out_file', dest='out_file', help='append solution to this file')

	args = parser.parse_args()
	g_files = args.files
	g_out_file = args.out_file

# Reads reel observations from the files in the parameter list.
# Returns the resulting list of observations.
def read_obs(*infiles):
	logging.info('READ_OBS from %s...', list(infiles))

	obs = list()

	for line in fileinput.input(*infiles):
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
	logging.info('READ_OBS DONE')

	return obs

# Prepares data structures: overlap matrix and reel graph
def setup():
	global g_obs
	global g_overlap
	# global g_graph

	logging.info('SETUP...')

	# prepare g_obs: remove duplicates to avoid both getting discarded as redundant
	g_obs = list(set(g_obs))
	g_obs.sort() # DEBUG: establish deterministic order of obs

	N = len(g_obs)
	elim_pieces = [] # redundant pieces list

	g_overlap = [[0] * N for i in range(0,N)]

	for i in range(0,N):
		for j in range(0,N):
			obs_i = g_obs[i]
			obs_j = g_obs[j]

			# mark redundant piece if we find any
			if (i != j) and (obs_i in obs_j):
				elim_pieces.append(i)

			g_overlap[i][j] = overlap(obs_i, obs_j)

	# eliminate initial pieces with complete overlap to reduce search space
	free_pieces = [i for i in range(0, len(g_obs)) if i not in elim_pieces]
	# g_graph = ReelGraph(free_pieces)

	logging.debug('g_obs is now %s (eliminated %s).', list(map(lambda x: g_obs[x], free_pieces)), elim_pieces)
	logging.debug('g_overlap = %s', g_overlap)

	logging.info('SETUP DONE')


# This is the main search algorithm.
# It operates on the global graph and returns the computed solution string.
def astar():
	global g_obs

	logging.info('ASTAR...')

	leaf = []   # heap of open ReelNodes which are leaves in the search graph -> paths left to explore

	# initialize leaf list with the root node
	# choose any obs as starting point for the solution
	free = list(range(1,len(g_obs)))
	cost = len(g_obs[0]) - g_overlap[0][0]
	node0 = ReelNode([0], free, cost)
	node0 = purify(node0)
	node0.est = est(node0) # NOTE: this is only useful for debug output because there is no other node to choose from at first

	leaf = [node0]

	# start of search
	cursor = heapq.heappop(leaf)
	logging.debug('Examine d=%s\tf(n)=%s\t%s\t%s)', len(cursor.sequence), cursor.est, solution(cursor.sequence), cursor)

	while(cursor.free):
		for s in successor(cursor):
			heapq.heappush(leaf, s)

		cursor = heapq.heappop(leaf)
		logging.debug('Examine d=%s\tf(n)=%s\t%s\t%s)', len(cursor.sequence), cursor.est, solution(cursor.sequence), cursor)

	logging.info('ASTAR DONE')
	return final_solution(cursor.sequence)

# Program entry point
def main():
	global g_out_file
	global g_files
	global g_obs

	handle_args()

	g_obs = read_obs(g_files)
	setup()
	result = astar()

	if g_out_file:
		out_file = io.open(g_out_file, 'a')
		out_file.write(result + '\n')
		out_file.close()
	else:
		sys.stdout.write(result + '\n')

if __name__ == "__main__":
	main()
