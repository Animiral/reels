#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

'''
The file README contains a general description of the program. Read on for implementation details.

reels.py is an A* search algorithm.

Nodes in the search graph consist of a partial solution built of one or more observed pieces and a list of free pieces, which are not yet represented in the solution.
In the start node, the partial solution contains only the largest single observed piece of the reel. All other pieces are free.
In the goal nodes, the solution is a complete sequence of observed pieces, some possibly overlapping with the next. The free list is empty.

To move from one node to the next, the program appends a free piece to the left or right of the partial solution.
The cost of the edge is the number of additional symbols in the partial solution after the operation.
Overlapping observations are therefore preferred to find a short solution.

To help with determining the cheapest path, the program uses an NxN matrix g_overlap, where N is the number of observed pieces.
g_overlap[i][j] is the number of overlapping symbols when appending g_obs[j] to the right of g_obs[i].

The heuristic function (estimated distance to goal) is the sum of minimum costs using g_overlap_left and g_overlap_right for every free piece.
'''

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

def overlap(left, right):
	'''Return the number of overlapping symbols when appending the right piece to the left piece.'''
	max_overlap = min(len(left),len(right)) - 1 # legal pieces can not overlap whole

	for i in range(max_overlap, 0, -1):
		if left[-i:] == right[:i]:
			return i
	else:
		return 0

def solution(sequence, context):
	'''Return the solution string from a list of obs indices representation.'''
	obs, overmat = context

	prev_index = sequence[0]
	S = obs[prev_index]

	for next_index in sequence[1:]:
		next_piece = obs[next_index]
		savings = overmat[prev_index][next_index]
		S += next_piece[savings:]
		prev_index = next_index

	return S

def final_solution(sequence, context):
	'''Return the final solution string from a list of obs indices representation.
	The difference to an intermediate solution is that due to its looped nature,
	overlap between the first and last piece can cut off some characters in the final solution.
	'''
	S = solution(sequence, context)

	loop_overlap = context.overmat[sequence[-1]][sequence[0]] # Careful: don’t repeat the start of the solution if it overlaps with the end piece
	if loop_overlap > 0:
		return S[:-loop_overlap]
	else:
		return S 

class ReelNode:
	'''The nodes in the search graph are partially solved reel problems.
	A node is impure if any one free piece is a proper substring of the partial solution.
	In that case, it should be purified before searching on.
	For two nodes n1 and n2, (not n1 < n2) and (not n1 > n2) is not equivalent to n1 == n2.
	This is deliberate because node equality and ordering are used for different, unrelated purposes.
	'''

	def __init__(self,s,f,c):
		self.sequence = s      # list of the observations included in the solution (in order)
		self.free = f          # set of free observations (also a list)
		self.cost = c          # path cost of the partial solution that is the sequence = number of symbols in solution(sequence).
		self.est = sys.maxsize # estimated total path cost through this node to the goal

	def __str__(self):
		'''String view for debugging: the cost and est is not important; we want to know the partial solution and free pieces'''
		return 'ReelNode({0},{1})'.format(self.sequence, self.free)

	def __lt__(self, other):
		'''Ordering for leaf heap.'''
		if self.est == other.est:
			return len(self.sequence) > len(other.sequence) # if tied, prefer to examine more complete solutions first
		else:
			return self.est < other.est

	# def __eq__(self, other):
	# 	'''Equality: used for the node hash table in ReelGraph.
	# 	Nodes may be re-discovered with lower est, but they must still go in the same hash bucket.
	# 	'''
	# 	return self.sequence == other.sequence and self.free == other.free

	# def __hash__(self):
	# 	'''Node hashing: XOR the bits of every item in the partial solution and the free list.
	# 	The bits are rotated around the integer after each XOR to make better use of the available space in the higher-order bits.
	# 	Partial solution and free list use different rotation phases so that moving the
	# 	first free piece to the end of the partial solution will produce a different hash.
	# 	'''
	# 	h = 0

	# 	for s in self.sequence:
	# 		h = ((h << 4) ^ (h >> 28)) ^ s

	# 	for f in self.free:
	# 		h = ((h << 5) ^ (h >> 27)) ^ s

	# 	return h

def est(node, context):
	'''Return the est of a node f(n) = g(n) + h(n), where g is the cost so far and h is the estimated cost to goal.
	g(n) is the number of symbols in the partial solution which are certainly a unique part of the final solution.
	Some possible overlapping symbols between the last and first piece of the partial solution are not counted
	towards the cost so far because even with some free pieces left over, the final reel might already be complete.
	h(n) is the estimated number of symbols that the free pieces are going to contribute towards the final reel.
	As an optimistic guess, we assume that every free piece will be placed such that it achieves optimal overlap
	with another piece to the left. The other piece could be any of the other free pieces or the tail of the sequence.
	Additionally, we discount a number of symbols according to the best possible overlap to the head of the sequence.
	'''
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

def successor(node, context):
	'''Generate all successors to the given node.'''
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
		succ.est = est(succ, context)

		yield succ

def handle_args():
	'''Parse and handle command arguments.'''
	parser = argparse.ArgumentParser(description='''
Reads input from FILES in order and writes a one-line result for each input file to OUTPUT.
If no input files are specified, reads from standard input.
If no output files are specified, writes to standard output.
		''')
	parser.add_argument('files', metavar='FILE', nargs='*', help='input file(s)')
	parser.add_argument('-o', '--out_file', help='append solution to this file')
	parser.add_argument('-a', '--algorithm', choices=['astar','dfs'], default='astar', help='search algorithm to use')
	parser.add_argument('-d', '--debug', action='store_true', default=False, help=argparse.SUPPRESS) #, help='debug log level')

	args = parser.parse_args()
	if args.debug: logging.basicConfig(level=logging.DEBUG)
	else:          logging.basicConfig(level=logging.WARNING)

	return args.files, args.out_file, args.algorithm

def make_obs(*in_files):
	'''Reads reel observations from the files in the parameter list.
	Returns the resulting list of observations.
	'''
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
	'''Construct the overlap matrix from the list of observations.
	The overlap matrix is an NxN array of arrays such that overmat[i][j] is the maximum number of symbols
	that can be overlapped when appending obs[j] to the right of obs[i].
	As a byproduct, this function notes a set of obs pieces which are substrings of some other piece and
	thus redundant. The members of the elim set are indexes into obs.
	The return value of make_overmat is (overmat, elim).
	'''
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

def setup(in_files):
	'''Prepare data structures for search: obs list, overlap matrix and free list for start node.'''
	logging.info('SETUP from %s...', list(in_files))

	obs = make_obs(in_files)
	overmat, elim = make_overmat(obs)
	free = [i for i in range(0, len(obs)) if i not in elim] # eliminate initial pieces with complete overlap to reduce search space
	context = Context(obs, overmat)

	# logging.debug('obs is now %s (eliminated %s).', list(map(lambda x: obs[x], free)), elim)
	# logging.debug('overmat = %s', overmat)

	logging.info('SETUP DONE')
	return free, context

def astar(root, context, goal_callback):
	'''This is the main search algorithm.
	It finds the optimal solution and calls goal_callback once with the goal as parameter.
	The goal node is also returned.
	'''
	logging.info('ASTAR...')

	leaf = [root] # heap of open ReelNodes which are leaves in the search graph -> paths left to explore

	# n_leaf = 100 # DEBUG counter for leaves
	# min_est = root.est
	# max_est = root.est

	# start of search
	cursor = heapq.heappop(leaf)
	# logging.debug('Examine d=%s\tf(n)=%s\t%s\t%s)', len(cursor.sequence), cursor.est, solution(cursor.sequence), cursor)

	while(cursor.free):
		for s in successor(cursor, context):
			heapq.heappush(leaf, s)
		# 	min_est = min(s.est, min_est) # DEBUG tracking
		# 	max_est = max(s.est, max_est) # DEBUG tracking

		# if len(leaf) > n_leaf:           # DEBUG report mem usage
		# 	logging.debug('len(leaf) = %s (d: %s ~ %s)', len(leaf), min_est, max_est)
		# 	n_leaf = len(leaf) * 1.5

		cursor = heapq.heappop(leaf)
		# logging.debug('Examine d=%s\tf(n)=%s\t%s\t%s)', len(cursor.sequence), cursor.est, solution(cursor.sequence), cursor)

	logging.info('ASTAR DONE')
	goal_callback(cursor)

	return cursor

def dfs(root, context, goal_callback, limit=sys.maxsize):
	'''An alternative greedy depth-first search algorithm.
	It produces solutions very quickly at first, but doesn’t offer the guarantee of an optimal solution.
	The program will keep running even after a solution has been found and keep producing better solutions,
	if it finds any, until the entire search space is exhausted.
	The optional limit parameter specifies a lower bound on the cost of a viable solution.
	Only goals below this limit are considered.
	'''

	# dummy goal, inferior to any actual goal found
	goal = type('DummyNode', (object,), {'est':limit})()
	goal.est = limit

	succ = sorted(successor(root, context))

	for s in succ:
		if s.free:
			if s.est < limit:
				goal = dfs(s, context, goal_callback, limit=goal.est)
		else:
			if s.est < goal.est:
				goal = s
				goal_callback(goal)

	return goal

def main():
	'''Program entry point.'''
	in_files, out_file, algorithm = handle_args()
	free, context = setup(in_files)
	search = getattr(sys.modules[main.__module__], algorithm)

	# Build root node
	# initialize leaf list with the root node
	# choose any obs as starting point for the solution
	cost = len(context.obs[free[0]]) - context.overmat[0][0]
	root = ReelNode([free[0]], free[1:], cost)
	root.est = est(root, context) # NOTE: this is only useful for debug output because there is no other node to choose from at first

	def print_goal_file(goal):
		solution_str = final_solution(goal.sequence, context)
		out_fd.write(solution_str + '\n')

	def print_goal_stdout(goal):
		solution_str = final_solution(goal.sequence, context)
		sys.stdout.write(solution_str + '\n')

	if out_file:
		out_fd = io.open(out_file, 'a')
		search(root, context, print_goal_file)
		out_fd.close()
	else:
		search(root, context, print_goal_stdout)

if __name__ == "__main__":
	main()
