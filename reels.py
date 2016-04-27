#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

'''
The file README contains a general description of the program. Read on for implementation details.

reels.py is an A* search algorithm.

Nodes in the search tree consist of a partial solution built of one or more observed pieces and a list of free pieces, which are not yet represented in the solution.
In the start node, the partial solution contains only one arbitrary observed piece of the reel. All other pieces are free.
In the goal nodes, the solution is a complete sequence of observed pieces, some possibly overlapping with the next. The free list is empty.

To move from one node to the next, the program appends a free piece to the right of the partial solution.
The cost of the edge is the number of additional symbols in the partial solution after the operation.
Overlapping observations are therefore preferred to find a short solution.

To help with determining the cheapest path, the program uses an NxN overlap matrix (overmat), where N is the number of observed pieces.
overmat[i][j] is the number of overlapping symbols when appending obs[j] to the right of obs[i].

The heuristic function (estimated distance to goal) is calculated assuming that that every piece can find a place in the final solution
with ideal overlap, i.e. as many symbols as possible overlap with the next piece.
'''

import sys
import logging

from heapq import heappush, heappop
from itertools import chain
from collections import namedtuple

# Context holds information about the search environment.
# obs is the list of observation strings taken from the source reel.
# overmat is the overlap matrix of precomputed overlaps between observations.
Context = namedtuple('Context', ['obs', 'overmat'])

# NOTE: Lists of observations are generally implemented as lists of indexes into the g_obs list.

class ReelNode:
	'''Represents one node in the search tree. Nodes are partially solved reel problems.

	node.sequence is the list of observed pieces, in order, that make up the partial solution.
	It is represented as a list of indices into the context.obs list.
	For example, the first piece contained in the partial solution is context.obs[node.sequence[0]].

	node.free is the list of observed pieces from the reel that are not yet in the partial solution.
	It is also represented as a list of indices, just like the sequence.
	To find a goal, the search must append free pieces to the sequence as efficiently as possible,
	exploiting the likelyhood that different observations cover overlapping parts of the reel.

	node.cost is the number of symbols in the partial solution which are certainly a unique part of the final solution.

	node.est is an optimistic guess at the number of symbols we can expect to see in the final reel,
	given the information from this node and the context. It is calculated in __calc_est().

	ReelNodes exist among a context of observations and pre-computed information about them (overmat).
	The node does not carry its own context pointer. It implements the flyweight pattern. The caller
	of a node’s methods must supply its context where necessary.
	'''

	def __calc_est(self, context):
		'''Return the estimated total solution cost f(n) = g(n) + h(n) for this node, where g is the cost so far
		and h is the estimated cost to goal.

		g(n) is the number of symbols in the partial solution which are certainly a unique part of the final solution.
		Some possibly overlapping symbols between the last and first piece of the partial solution are not counted
		towards the cost so far because even with some free pieces left over, the final reel might already be complete.

		h(n) is the estimated number of symbols that the free pieces are going to contribute towards the final reel.
		As an optimistic guess, we assume that every free piece will be placed such that it achieves optimal overlap
		with another piece to the left. The other piece could be any of the other free pieces or the tail of the sequence.

		Additionally, we discount a number of symbols according to the best possible overlap to the head of the sequence.
		'''
		obs, overmat = context

		G = self.cost

		free = self.free
		if not free: return G   # this is goal node

		A = self.sequence[0]
		Z = self.sequence[-1]

		H = overmat[Z][A]                            # revert finished-loop assumption from cost g(n)
		H -= max(map(lambda a: overmat[a][A], free)) # assume best overlap from any free piece to A

		for f in free:
			free_Z = chain(free, [Z])
			max_overlap = max(map(lambda a: overmat[a][f], free_Z))
			H += len(obs[f]) - max_overlap

		H = max(0,H)

		return G + H

	def __init__(self, sequence, free, cost, context):
		self.sequence = sequence            # list of the observations included in the solution (in order)
		self.free = free                    # set of free observations (also a list)
		self.cost = cost                    # path cost of the partial solution that is the sequence = number of symbols in self.__solution().
		self.est = self.__calc_est(context) # estimated total path cost through this node to the goal

	def __str__(self):
		'''String view for debugging: the cost and est is not important; we want to know the partial solution and free pieces'''
		return 'ReelNode({0},{1})'.format(self.sequence, self.free)

	def __lt__(self, other):
		'''Ordering for leaf heap.'''
		if self.est == other.est:
			return len(self.sequence) > len(other.sequence) # if tied, prefer to examine more complete solutions first
		else:
			return self.est < other.est

	def __solution(self, context):
		'''Return the partial solution string from a list of obs indices representation.'''
		obs, overmat = context

		prev_index = self.sequence[0]
		S = obs[prev_index]

		for next_index in self.sequence[1:]:
			next_piece = obs[next_index]
			savings = overmat[prev_index][next_index]
			S += next_piece[savings:]
			prev_index = next_index

		return S

	def final_solution(self, context):
		'''Return the final solution string from a list of obs indices representation.
		The difference to an intermediate solution is that due to its looped nature,
		overlap between the first and last piece can cut off some characters in the final solution.
		'''
		S = self.__solution(context)

		loop_overlap = context.overmat[self.sequence[-1]][self.sequence[0]] # Careful: don’t repeat the start of the solution if it overlaps with the end piece
		if loop_overlap > 0:
			return S[:-loop_overlap]
		else:
			return S

	def successor(self, context):
		'''Generate all successors to this node.'''
		obs, overmat = context
		sequence = self.sequence
		free = self.free
		cost = self.cost

		overlap = 0

		for i in range(0, len(free)):
			P = free[i]       # new piece
			A = sequence[0]   # sequence first piece
			Z = sequence[-1]  # sequence last piece

			# append free piece after partial solution
			succ_sequence = sequence + [P]
			succ_free = free[:i] + free[i+1:]
			succ_cost = cost + len(obs[P]) + overmat[Z][A] - overmat[Z][P] - overmat[P][A]

			succ = ReelNode(succ_sequence, succ_free, succ_cost, context)
			yield succ

# ------------------------------- end of class ReelNode ------------------------------- #

def make_obs(*in_files):
	'''Read reel observations from the files in the parameter list.
	Return the resulting list of observations.
	'''
	import fileinput
	import re

	obs = []

	for line in fileinput.input(*in_files):
		line = line.strip() # remove newline at the end
		if not line: continue # ignore empty lines

		# lines are invalid if they contain non-word characters (invalid symbols)
		if not re.match('^\w+$', line):
			raise RuntimeError('Illegal symbol in "{0}"'.format(line))

		obs.append(line)

	if not obs: raise RuntimeError('No input was given!')
	
	obs = list(set(obs)) # remove duplicates (avoids both getting discarded as redundant later)
	obs.sort() # DEBUG: establish deterministic order of obs

	return obs

def overlap(left, right):
	'''Return the number of overlapping symbols when appending the right piece to the left piece.'''
	max_overlap = min(len(left),len(right)) - 1 # legal pieces can not overlap whole

	for i in range(max_overlap, 0, -1):
		if left[-i:] == right[:i]:
			return i
	else:
		return 0

def make_overmat(obs):
	'''Construct the overlap matrix from the list of observations.
	The overlap matrix is an NxN array of arrays such that overmat[i][j] is the maximum number of symbols
	that can be overlapped when appending obs[j] to the right of obs[i].
	As a byproduct, this function notes a set of obs pieces which are substrings of some other piece and
	thus redundant. The members of the elim set are indices into obs.
	The return value of make_overmat is (overmat, elim).
	'''

	N = len(obs)
	elim = set() # redundant pieces

	overmat = [[0] * N for i in range(0,N)] # overlap matrix

	for i in range(0,N):
		for j in range(0,N):
			if (i != j) and (obs[i] in obs[j]):
				elim.add(i) # mark redundant piece if we find any
			overmat[i][j] = overlap(obs[i], obs[j])

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
	cursor = heappop(leaf)

	while(cursor.free):
		for s in cursor.successor(context):
			heappush(leaf, s)

		cursor = heappop(leaf)

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

	succ = sorted(root.successor(context))

	for s in succ:
		if s.free:
			if s.est < limit:
				goal = dfs(s, context, goal_callback, limit=goal.est)
		else:
			if s.est < goal.est:
				goal = s
				goal_callback(goal)

	return goal

def handle_args():
	'''Parse and handle command arguments.'''
	import argparse

	parser = argparse.ArgumentParser(description='''
		Reads input from FILES in order and writes a one-line result for each input file to OUTPUT.
		If no input files are specified, reads from standard input.
		If no output files are specified, writes to standard output.
		''')
	parser.add_argument('files', metavar='FILE', nargs='*', help='input file(s)')
	parser.add_argument('-o', '--out_file', help='append solution to this file')
	parser.add_argument('-a', '--algorithm', choices=['astar','dfs'], default='astar', help='search algorithm to use')
	parser.add_argument('-d', '--debug', action='store_true', default=False, help=argparse.SUPPRESS)

	args = parser.parse_args()
	if args.debug: logging.basicConfig(level=logging.DEBUG)
	else:          logging.basicConfig(level=logging.WARNING)

	return args.files, args.out_file, args.algorithm

def main():
	'''Program entry point.'''
	import io

	in_files, out_file, algorithm = handle_args()
	free, context = setup(in_files)
	search = getattr(sys.modules[main.__module__], algorithm)

	# Build root node
	# choose any obs as starting point for the solution
	cost = len(context.obs[free[0]]) - context.overmat[0][0]
	root = ReelNode([free[0]], free[1:], cost, context)

	def print_goal_file(goal):
		solution_str = goal.final_solution(context)
		out_fd.write(solution_str + '\n')

	def print_goal_stdout(goal):
		solution_str = goal.final_solution(context)
		sys.stdout.write(solution_str + '\n')

	if out_file:
		out_fd = io.open(out_file, 'a')
		search(root, context, print_goal_file)
		out_fd.close()
	else:
		search(root, context, print_goal_stdout)

if __name__ == "__main__":
	main()
