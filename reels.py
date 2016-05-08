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
import functools

from heapq import heappush, heappop
from itertools import chain
from collections import namedtuple

# Context holds information about the search environment.
# obs is the list of observation strings taken from the source reel.
# overmat is the overlap matrix of precomputed overlaps between observations.
# pref is the list of preferred overlaps for every free piece.
Context = namedtuple('Context', ['obs', 'overmat', 'pref'])

def trace(func):
	'''Decorator which outputs name of called function to log'''
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		logging.info('ENTER %s', func.__name__)
		result = func(*args, **kwargs)
		logging.info('EXIT %s', func.__name__)
		return result

	return wrapper

def is_subsequence(haystack, needle):
	'''Return True if the list of symbols in needle also occurs in the haystack (also a list of symbols).
	This is the most naive, but also concise implementation. It only runs in setup() on generally short lists.'''
	for i in range(len(haystack) - len(needle) + 1):
		for j in range(len(needle)):
			if haystack[i+j] != needle[j]:
				break
		else:
			return True
	else:
		return False

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
	# __slots__ = ['sequence','free','cost','est']   # python reports sys.getsizeof(ReelNode) = 72 with __slots__, 56 without __slots__

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
		obs, overmat, pref = context

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
		return '<<{0}: {1},{2}>>'.format(self.est, self.sequence, self.free)

	def __lt__(self, other):
		'''Ordering for leaf heap.'''
		if self.est == other.est:
			return len(self.sequence) > len(other.sequence) # if tied, prefer to examine more complete solutions first
		else:
			return self.est < other.est

	def __solution(self, context):
		'''Return the partial solution string from a list of obs indices representation.'''
		import copy

		obs, overmat, pref = context

		prev_index = self.sequence[0]
		S = copy.deepcopy(obs[prev_index])

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
		obs, overmat, pref = context
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

def uniq(sorted_list):
	'''custom uniq function to replace python’s set(), which does not work on lists of lists'''
	n = len(sorted_list)
	if 0 == n: return
	p = sorted_list[0]
	yield p
	i = 1
	while i < n:
		q = sorted_list[i]
		if p != q:
			yield q
			p = q
		i = i + 1

@trace
def read_obs(in_file):
	'''Read reel observations from the in_file.
	Return the resulting list of observations, sorted and duplicates removed.

	In this default implementation, every character on every line is a symbol.
	There are no separators.
	'''
	import fileinput
	import re

	obs = []

	for line in fileinput.input(in_file):
		line = line.strip() # remove newline at the end
		if not line: continue # ignore empty lines

		# lines are invalid if they contain non-word characters (invalid symbols)
		if not re.match('^\w+$', line):
			raise RuntimeError('Illegal symbol in "{0}"'.format(line))

		obs.append(list(line))

	if not obs: raise RuntimeError('No input was given!')
	obs = list(uniq(sorted(obs))) # remove duplicates (avoids both getting discarded as redundant later)
	# obs = list(set(obs)) # TypeError: 'list' objects are unhashable
	return obs

def read_obs_csv(in_file, dialect):
	'''Read reel observations from the in_file using the specified CSV dialect.
	Return the resulting list of observations, sorted and duplicates removed.

	The input format is CSV, where each row specifies one observation and each
	value in the row is a symbol name. Symbols can thus have multi-character names.
	'''
	import csv

	obs = []
	_open_fd = lambda: open(in_file, newline='') if in_file else sys.stdin

	with _open_fd() as fd:
		if not dialect:
			dialect='excel'
		# NOTE: Sniffer mistakenly detects '\r' instead of ',' as delimiter
		# 	dialect = csv.Sniffer().sniff(fd.read(1024))
		# 	fd.seek(0)
		reader = csv.reader(fd, dialect, strict=True)

		for row in reader:
			obs.append(row)

	if not obs: raise RuntimeError('No input was given!')
	obs = list(uniq(sorted(obs))) # remove duplicates (avoids both getting discarded as redundant later)
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

	overmat = [[None] * N for i in range(0,N)] # overlap matrix

	for i in range(0,N):
		for j in range(0,N):
			if (i != j) and is_subsequence(obs[j], obs[i]):
				elim.add(i) # mark redundant piece if we find any
				break
			else:
				overmat[i][j] = overlap(obs[i], obs[j])

	return overmat, elim

def make_pref(obs, overmat, free):
	'''Construct the pref list from the list of observations and their overlaps.
	The pref list is a list which, for every free obs piece, gives the list of
	other pieces that this piece would prefer overlapping with.

	pref[i][0] is thus the obs index for the piece i such that
	overmat[i][pref[i][0]] >= overmat[i][pref[i][x]], x > 0.

	The order of the returned list is the same as the obs list from the obs parameter.
	'''
	def single_pref(i):
		other_free = filter(lambda a: a != i, free)
		best_overlap_first = sorted(other_free, key=lambda f: overmat[f][i])
		return list(best_overlap_first) 

	return [ single_pref(i) for i in range(len(obs)) ]	

@trace
def setup(in_file, read_obs_func):
	'''Prepare data structures for search: obs list, overlap matrix and free list for start node.'''

	obs = read_obs_func(in_file)
	logging.debug('obs = \n%s', '\n'.join(map(lambda x: '[{0}] '.format(x) + ' '.join(map(str,obs[x])), range(len(obs)))))
	overmat, elim = make_overmat(obs)
	obs = [(None if i in elim else obs[i]) for i in range(len(obs))] # eliminate initial pieces with
	free = [i for i in range(len(obs)) if i not in elim]             # complete overlap to reduce search space
	pref = make_pref(obs, overmat, free)
	context = Context(obs, overmat, pref)

	logging.debug('obs is now \n%s\n(eliminated %s).', '\n'.join(map(lambda x: '[{0}] '.format(x) + ' '.join(map(str,obs[x])), free)), elim)
	logging.debug('overmat =\n%s', '\n'.join(map(lambda line: '  '.join(map(str,line)), overmat)))

	return free, context

@trace
def astar(root, context, goal_callback, sym_limit, full, beat_func):
	'''This is the main search algorithm.
	It finds the optimal solution and calls goal_callback once with the goal as parameter.
	The goal node is also returned.

	The sym_limit parameter specifies an upper bound on the cost of a viable solution.
	Only goals below this limit are considered.

	solutions is the maximum number of solutions to produce. In the case of the A* algorithm,
	it produces just one solution anyway, unless full is True.

	If full is set to True, the search will return all qualified solutions instead of just one.
	'''
	leaves = [root] # heap of open ReelNodes which are leaves in the search graph -> paths left to explore
	cursor = heappop(leaves)
	quit = False

	examined = 0   # DEBUG: examined nodes counter
	discovered = 1 # DEBUG: discovered nodes counter (1 for root)
	memorized = 1  # DEBUG: memorized nodes counter

	while cursor.est <= sym_limit and not quit:
		beat_func(len(leaves)) # check in with caller

		logging.debug('Examine %s', cursor)
		examined = examined + 1

		if cursor.free:
			for s in cursor.successor(context):
				discovered = discovered + 1
				if s.est <= sym_limit:
					memorized = memorized + 1
					heappush(leaves, s)
		else:
			quit = not goal_callback(cursor)
			if quit or not full: break # one solution is enough
			sym_limit = cursor.est

		try:
			cursor = heappop(leaves)
		except IndexError:
			quit = True # no more goals available

	return examined, discovered, memorized

@trace
def dfs(root, context, goal_callback, sym_limit, full, beat_func):
	'''An alternative greedy depth-first search algorithm.
	It produces solutions very quickly at first, but doesn’t offer the guarantee of an optimal solution.
	The program will keep running even after a solution has been found and keep producing better solutions,
	if it finds any, until the entire search space is exhausted.
	'''
	from operator import lt, le

	# dummy goal, inferior to any actual goal found
	goal = type('DummyNode', (object,), {'est':sym_limit})()
	goal.est = sym_limit
	op = le

	if not full:
		op = lt
		goal.est = goal.est + 1 # fix off-by-one for user-specified sym_limit

	def _dfs(root, goal):
		'''recursive depth search implementation'''
		beat_func() # check in with caller

		succ = sorted(root.successor(context))
		quit = False
		examined = 1   # DEBUG: examined nodes counter
		discovered = 0 # DEBUG: discovered nodes counter

		for s in succ:
			discovered = discovered + 1
			if op(s.est, goal.est):
				if s.free:
					goal, quit, ex, disc = _dfs(s, goal)
					examined = examined + ex
					discovered = discovered + disc
				else:
					goal = s
					examined = examined + 1
					quit = not goal_callback(goal)

			if quit: break

		return goal, quit, examined, discovered

	_, _, examined, discovered = _dfs(root, goal)
	return examined, discovered, examined

def handle_args(argv=None):
	'''Parse and handle command arguments.'''
	import argparse
	import csv

	parser = argparse.ArgumentParser(description='''
		Reads input from FILE and writes a one-line result to out_file.
		If no input file is specified, reads from standard input.
		If no output file is specified, writes to standard output.
		''')
	parser.add_argument('in_file', metavar='FILE', type=str, nargs='?', help='input file')
	parser.add_argument('-o', '--out_file', help='append solution to this file')
	parser.add_argument('-a', '--algorithm', choices=['astar','dfs'], default='astar', help='search algorithm to use')
	parser.add_argument('--csv', action='store_true', default=False, help='specify default input format as CSV')
	parser.add_argument('-d', '--dialect', type=str, help='CSV dialect ({0})'.format(','.join(csv.list_dialects())))
	parser.add_argument('-n', '--solutions', type=int, default=sys.maxsize, help='halt after at most n solutions')
	parser.add_argument('-l', '--sym-limit', type=int, default=sys.maxsize, help='upper boundary for number of symbols in a solution')
	parser.add_argument('-f', '--full', action='store_true', default=False, help='do a full search for all, not just one, shortest solution')
	parser.add_argument('-t', '--timeout', type=int, default=None, help='time limit in seconds for search')
	parser.add_argument('-m', '--memsize', type=int, default=None, help='search node count limit for the process')
	parser.add_argument('-e', '--debug', action='store_true', default=False, help=argparse.SUPPRESS)   # activate debug log level
	parser.add_argument('-x', '--print-node-count', action='store_true', default=False, help=argparse.SUPPRESS)  # instead of output, print only examined-nodes,discovered-nodes

	args = parser.parse_args(argv)
	if args.debug: logging.basicConfig(level=logging.DEBUG)
	else:          logging.basicConfig(level=logging.WARNING)

	if args.in_file and args.in_file.endswith('.csv'): # special case: if file ext indicates CSV, always parse CSV
		args.csv = True

	return args

def run(free, context, search, sym_limit, full, solutions, out_fd, format_solution, timeout, memsize, debug_print_node_count=False):
	'''Runs the search algorithm with the given configuration from the arguments.'''
	import time

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
			sys.exit(1)

		if beat.memsize and node_count > beat.memsize:
			logging.error('Search exceeded the memory limit of %s nodes.', beat.memsize)
			sys.exit(1)

		# DEBUG: hier weiter, memsize checken
		# Test impl of beat() in search algos
		# Add beat_func to search call in profile_reels.py
		# Update docstrings in search funcs, README on return codes & parameters

	print_goal.print_count = solutions
	if debug_print_node_count:
		import io
		debug_print_fd = out_fd
		out_fd = io.open('/dev/null','a')
	# print_func = (lambda goal: True) if debug_print_node_count else print_goal 

	if timeout:
		beat.cutoff_time = time.time() + timeout

	beat.timeout = timeout
	beat.memsize = memsize

	# Build root node
	# choose any obs as starting point for the solution
	cost = len(context.obs[free[0]]) - context.overmat[0][0]
	root = ReelNode([free[0]], free[1:], cost, context)

	with out_fd: # close file when finished, come what may (exceptions etc)
		examined, discovered, memorized = search(root, context, print_goal, sym_limit, full, beat)

	if debug_print_node_count:
		debug_print_fd.write('{0},{1},{2}\n'.format(examined, discovered, memorized))

	logging.debug('Examined %s nodes.', examined)
	logging.debug('Discovered %s nodes.', discovered)
	logging.debug('Memorized %s nodes.', memorized)

def main():
	'''Program entry point.'''
	import io
	import functools

	args = handle_args()
	read_obs_func = functools.partial(read_obs_csv, dialect=args.dialect) if args.csv else read_obs
	free, context = setup(args.in_file, read_obs_func)
	search = {'astar': astar, 'dfs': dfs} [args.algorithm]
	out_fd = io.open(args.out_file, 'a') if args.out_file else sys.stdout
	format_solution = (lambda s: ','.join(s)) if args.csv else (lambda s: ''.join(s))

	run(free, context, search, args.sym_limit, args.full, args.solutions, out_fd, format_solution, args.timeout, args.memsize, args.print_node_count)
	return 0

if __name__ == "__main__":
	main()
