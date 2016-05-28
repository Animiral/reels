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
import itertools
import copy

from heapq import heappush, heappop
from collections import namedtuple
from operator import itemgetter

# ReelContext holds information about the search environment.
# obs is the list of observation strings taken from the source reel.
# overmat is the overlap matrix of precomputed overlaps between observations.
# pref is the list of preferred overlaps for every free piece.
# ReelContext = namedtuple('ReelContext', ['obs', 'overmat', 'pref'])
ReelContext = namedtuple('ReelContext', ['obs', 'free', 'overmat', 'pref'])
EstContext = namedtuple('EstContext', ['overmat', 'pref', 'lefts', 'A', 'Z'])

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

def raise_assoc(assoc, index, context):
	'''Associate the free piece or Z piece at index with the nearest available counterpart according to its pref.'''
	_, pref, lefts, A, Z = context

	pref_i = pref[index]
	a = assoc[index]

	if a >= len(pref_i):
		return False

	while True:
		# abort search if there is no valid right-piece to raise to
		if a >= len(pref_i):
			logging.debug('raise_assoc failed for index %s', index)
			return False

		# Check for suitability of right-piece. Valid right-pieces are the same as the left pieces (free pieces in general),
		# except that Z is an exclusive left piece and A is an exclusive right piece, and A is not a valid right piece for Z.
		rhs = pref_i[a] 
		if index == Z: rhs_valid = rhs in lefts
		else:          rhs_valid = rhs == A or (rhs != Z and rhs in lefts)

		if rhs_valid: 
			break

		a += 1

	assoc[index] = a
	return True

class EstNode:
	'''Node for searching the estimation heuristic.

	The estimated distance to a complete solution initially assumes that every left-hand piece will get its
	ideal match from the right-hand pieces for maximum overlap.
	This estimate often contains conflicts (duplicates) in its left-right associations.

	To improve on the estimate, we attempt to associate some left-hand pieces with other, less ideal
	right-hand pieces until we arrive at a conflict-free assoc.

	The search for this assoc is a breadth-first search. In every step, we resolve the conflicts in the
	current assoc by advancing along every piece’s pref list. The cost of the search nodes is the number
	of potentially overlapped symbols that had to be sacrificed in favor of the better estimate.

	The context that must be passed to every EstNode operation is an EstContext namedtuple.
	'''

	def __init__(self, assoc, cost, context):
		overmat, pref, lefts, _, _ = context
		self.assoc = assoc
		self.cost = cost

	def __lt__(self, other):
		'''Ordering for leaf heap.'''
		return self.cost < other.cost

	def __find_conflicts(self, context):
		'''Return every conflict in self.assoc.

		A conflict is a left-hand to right-hand association such that
		pref[p][self.assoc[p]] == pref[q][self.assoc[q]], p != q, where p and q are both left-hand pieces
		(free pieces + Z).

		The representation of one such conflict is a list of all the left-hands (lhs) involved.
		This method returns an iterable over all the detected conflicts.
		'''
		_, pref, lefts, _, _ = context
		rhs_lhs = ((pref[p][self.assoc[p]], p) for p in context.lefts)
		rhs_lhs = sorted(rhs_lhs) # groupby requires sorted input
		# rhs_lhs.sort()

		for rhs, pair_iter in itertools.groupby(rhs_lhs, key=itemgetter(0)):
			lhs = list(map(itemgetter(1), pair_iter))
			if len(lhs) > 1: # more than one lhs associated with the rhs is a conflict
				yield lhs

	def __resolutions(self, conflict):
		'''Resolving a conflict involves nothing more than re-associating all but one of the involved
		left-hand pieces. One resolution is a list of the pieces to be re-associated. For a conflict
		with N involved pieces, there are thus N resolutions.
		'''
		for i in range(len(conflict)):
			yield conflict[:i] + conflict[i+1:]

	def expand(self, context):
		'''Evaluate the finer details of this node (conflicts) and test the goal condition.
		If this node is a goal, return False.
		If this node has conflicts and is thus not a goal, return True.
		'''
		_, pref, lefts, _, _ = context

		self.conflicts = list(self.__find_conflicts(context))
		return bool(self.conflicts)

		# Short goal test
		# rhs = (pref[p][self.assoc[p]] for p in lefts) # sequence of all currently assigned right pieces
		# rhs_dup = list(rhs)
		# rhs_uniq = set(rhs_dup)

		# if len(rhs_dup) == len(rhs_uniq): # every right-hand piece is uniquely assigned (no conflicts)
		# 	return False
		# else:
		# 	return True	

	def successor(self, context):
		'''Generate all successors to this EstNode in the estimate search tree.'''
		overmat, pref, _, _, _ = context
		conflicts = self.__find_conflicts(context)
		resolutions = list(map(self.__resolutions, conflicts)) # for each conflict, we now have a list of possible resolutions for that conflict.

		# Every successor node emerges from a plan.
		# A plan is a collection of left-hand pieces that should be re-associated so that they might yield a conflict-free estimate.
		# A plan is generated by attempting one (of several possible) resolutions for every conflict instance.
		for plan in itertools.product(*resolutions):
			succ_assoc = copy.deepcopy(self.assoc)
			succ_cost = self.cost
			valid = True

			# advance is the piece index of one piece which we must re-associate by advancing its assoc entry to the next pref.
			for advance in itertools.chain(*plan):
				succ_assoc[advance] += 1
				valid = valid and raise_assoc(succ_assoc, advance, context)
				if not valid: break

				# increase succ cost by # of overlap lost
				pref_row = pref[advance]
				self_pref = pref_row[self.assoc[advance]]
				succ_pref = pref_row[succ_assoc[advance]]
				self_overlap = overmat[advance][self_pref] 
				succ_overlap = overmat[advance][succ_pref] 
				loss = self_overlap - succ_overlap
				assert loss >= 0
				succ_cost += loss

			if valid:
				yield EstNode(succ_assoc, succ_cost, context)

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

		Neat heuristic:
		A mapping (l,r) of left-pieces to right-pieces is a complete list that associates with each free piece in the node
		plus the Z piece (end of the partial solution) another free piece or A (start of the partial solution). Each
		left-piece and each right-piece must appear exactly once in the mapping, because it is impossible to append to
		the same piece more than once. (Z,A) is not a valid association (because it would skip the remaining free pieces).
		To compute the optimistic heuristic h(n), we attempt to find among the possible mappings the one that provides
		the best possible overlap between all left-pieces and right-pieces.
		The search for this best mapping starts with the pref list, giving each piece the ideal counterpart. From there,
		conflicting assignments are advanced down the pref list until a valid mapping emerges.

		Basic heuristic:
		As an optimistic guess, we assume that every free piece will be placed such that it achieves optimal overlap
		with another piece to the left. The other piece could be any of the other free pieces or the tail of the sequence.

		Additionally, we discount a number of symbols according to the best possible overlap to the head of the sequence.
		'''
		import functools

		# obs, overmat, pref = context
		obs, _, overmat, pref = context
		# sequence, free = self.sequence, self.free
		sequence, free = self.__sequence_free(context)

		G = self.cost

		if not free: return G   # this is goal node

		A = sequence[0]
		Z = sequence[-1]

		# ~~~~~~~~~~~~~~~~~~~~~~ neat heuristic ~~~~~~~~~~~~~~~~~~~~~~

		# enum_obs = list(range(len(obs)))
		# lefts = free + [Z] # left-hand pieces of association heuristic
		# assoc = [0 if x in free or x == Z else None for x in enum_obs] # root assoc for heuristic search

		# self.est = 0 # DEBUG: set temp value
		# logging.debug('est %s', self)
		# logging.debug('assoc for obs=%s, pref=%s = %s', obs, pref, assoc)

		# def calc_over(lefts, overmat, pref, assoc):
		# 	'''Get total overlap from the assoc configuration.'''
		# 	over = 0

		# 	for i in lefts:
		# 		a = assoc[i]      # how many steps down in the preference this piece had to go
		# 		p = pref[i][a]    # piece which is attached to i according to our configuration
		# 		o = overmat[i][p] # overlapping symbols count
		# 		over += o

		# 	return over

		# context = EstContext(overmat, pref, lefts, A, Z)

		# # update assoc to point to valid pieces
		# for i in lefts:
		# 	raise_assoc(assoc, i, context)

		# root = EstNode(assoc, 0, context)
		# leaves = [root] # bfs node heap
		# clean_config = None # bfs goal

		# while leaves:
		# 	node = heappop(leaves)
		# 	logging.debug('Est examine <<%s>>', node.assoc)	

		# 	if not node.expand(context): # this is goal
		# 		clean_config = node.assoc
		# 		over = calc_over(lefts, overmat, pref, node.assoc)
		# 		break

		# 	for succ in node.successor(context):
		# 		heappush(leaves, succ)

		# 	# DEBUG: just one iteration	
		# 	# over = calc_over(lefts, overmat, pref, node.assoc)
		# 	# break

		# H = overmat[Z][A]                                 # revert finished-loop assumption from cost g(n)
		# H += functools.reduce(lambda a, f: a + len(obs[f]), free, 0)   # total length of leftover pieces without overlap
		# H -= over

		# logging.debug('G=%s, H=%s, over=%s, est=%s', G, H, over, G+H)

		# ~~~~~~~~~~~~~~~~~~~~~~ basic heuristic ~~~~~~~~~~~~~~~~~~~~~~
		H = overmat[Z][A]                            # revert finished-loop assumption from cost g(n)
		H -= max(map(lambda a: overmat[a][A], free)) # assume best overlap from any free piece to A

		for f in free:
			free_Z = itertools.chain(free, [Z])
			max_overlap = max(map(lambda a: overmat[a][f], free_Z))
			H += len(obs[f]) - max_overlap
		# ~~~~~~~~~~~~~~~~~~~~~~ end basic heuristic ~~~~~~~~~~~~~~~~~~~~~~

		H = max(0,H)

		# DEBUG: this will happen anyway, but not if the debug stmt above (set est=0) happened
		self.est = G + H

		return G + H

	def __init__(self, parent, piece, cost, context):
		# self.sequence = sequence            # list of the observations included in the solution (in order)
		# self.free = free                    # set of free observations (also a list)
		self.parent = parent                # parent node in the search tree
		self.piece = piece                  # last piece in the partial solution up to this node
		self.cost = cost                    # path cost of the partial solution that is the sequence = number of symbols in self.__solution().
		self.est = self.__calc_est(context) # estimated total path cost through this node to the goal

	def __str__(self):
		'''String view for debugging: the cost and est is not important; we want to know the partial solution and free pieces'''
		sequence, _ = self.__sequence_free(ReelContext([],[],[],[])) # placeholder context
		# sequence, _ = self.__sequence_free(ReelContext())
		return '<<{0}: {1}>>'.format(self.est, sequence)

	def __lt__(self, other):
		'''Ordering for leaf heap.'''
		if self.est == other.est:
			return self.__len() > other.__len() # if tied, prefer to examine more complete solutions first
		else:
			return self.est < other.est

	# Helper functions for new representation:
	# Every ReelNode just remembers its parent and its last piece and not the whole partial solution.
	# The missing data can be reconstructed using these helpers.
	def __len(self):
		'''Returns the number of pieces in this node’s partial solution.'''
		# return len(self.sequence) 
		parent = self.parent
		self_len = 1
		# while parent:
		# 	parent = parent.parent
		# 	self_len += 1
		if parent:
			self_len += len(parent.sequence)

		return self_len

	def __sequence_free(self, context):
		'''Reconstruct and return the partial solution sequence and the free list
		from this node’s and its parent’s piece information.
		'''
		if self.parent:
			# sequence = self.parent.__sequence()
			# sequence = copy.deepcopy(self.parent.sequence)
			sequence = self.parent.sequence + [self.piece]
		else:
			# sequence = []
			sequence = [self.piece]

		# sequence.append(self.piece)
		free = list(set(context.free) - set(sequence))

		return sequence, free

	def __solution(self, sequence, context):
		'''Return the partial solution string from a list of obs indices representation.'''
		# obs, overmat, pref = context
		obs, _, overmat, pref = context

		prev_index = sequence[0]
		S = copy.deepcopy(obs[prev_index])

		for next_index in sequence[1:]:
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
		sequence = self.sequence
		# sequence, _ = self.__sequence_free(context)
		S = self.__solution(sequence, context)

		loop_overlap = context.overmat[sequence[-1]][sequence[0]] # Careful: don’t repeat the start of the solution if it overlaps with the end piece
		if loop_overlap > 0:
			return S[:-loop_overlap]
		else:
			return S

	def expand(self, context):
		'''Evaluate the finer details of this node (sequence, free) and test the goal condition.
		If this node is a goal, return False.
		If this node has free pieces and is thus not a goal, return True.
		'''
		self.sequence, self.free = self.__sequence_free(context)
		return bool(self.free)

	def successor(self, context):
		'''Generate all successors to this node.'''
		obs, _, overmat, pref = context
		sequence = self.sequence
		free = self.free
		# sequence, free = self.__sequence_free(context)
		cost = self.cost

		overlap = 0

		for i, p in enumerate(free):
			P = free[i]       # new piece
			A = sequence[0]   # sequence first piece
			Z = sequence[-1]  # sequence last piece

			# append free piece after partial solution
			# succ_sequence = sequence + [P]
			# succ_free = free[:i] + free[i+1:]
			succ_cost = cost + len(obs[P]) + overmat[Z][A] - overmat[Z][P] - overmat[P][A]
			# succ = ReelNode(succ_sequence, succ_free, succ_cost, context)
			succ = ReelNode(self, P, succ_cost, context)
			yield succ

# ------------------------------- end of class ReelNode ------------------------------- #

def uniq(sorted_list):
	'''Custom uniq function to replace python’s set(), which does not work on lists of lists.
	This implementation was lifted from the recipee unique_justseen at https://docs.python.org/3/library/itertools.html'''
	return map(next, map(itemgetter(1), itertools.groupby(sorted_list)))
	# n = len(sorted_list)
	# if 0 == n: return
	# p = sorted_list[0]
	# yield p
	# i = 1
	# while i < n:
	# 	q = sorted_list[i]
	# 	if p != q:
	# 		yield q
	# 		p = q
	# 	i = i + 1

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

def read_obs_csv(in_file, dialect='excel'):
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
		best_overlap_first = sorted(other_free, key=lambda f: -overmat[i][f])
		return list(best_overlap_first) 

	return [ single_pref(i) if i in free else [] for i in range(len(obs)) ]	

@trace
def setup(in_file, read_obs_func):
	'''Prepare data structures for search: obs list, overlap matrix and free list for start node.'''

	obs = read_obs_func(in_file)
	logging.debug('obs = \n%s', '\n'.join(map(lambda x: '[{0}] '.format(x) + ' '.join(map(str,obs[x])), range(len(obs)))))
	overmat, elim = make_overmat(obs)
	obs = [(None if i in elim else obs[i]) for i in range(len(obs))] # eliminate initial pieces with
	free = [i for i in range(len(obs)) if i not in elim]             # complete overlap to reduce search space
	pref = make_pref(obs, overmat, free)
	context = ReelContext(obs, free, overmat, pref)

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

		if cursor.expand(context): # not goal, search goes on
			for s in cursor.successor(context):
				discovered = discovered + 1
				if s.est <= sym_limit:
					memorized = memorized + 1
					heappush(leaves, s)
		else: # goal
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
				if s.expand(context):
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
	parser.add_argument('-e', '--debug', action='store_true', default=False, help=argparse.SUPPRESS)   # activate debug log level (distinct from interpreter -O option, which affects asserts)
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
	piece0 = free[0] # choose not-eliminated piece
	cost = len(context.obs[piece0]) - context.overmat[piece0][piece0]
	# root = ReelNode([free[0]], free[1:], cost, context)
	root = ReelNode(None, piece0, cost, context)

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
