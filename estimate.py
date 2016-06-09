#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

'''
This is the distance-to-goal heuristic component of the reels.py application.
The file README contains a general description of the program. Read on for implementation details.

The heuristic function (estimated distance to goal) for a partial reel solution is calculated using its own tree search algorithm.
The estimate search attempts to fit the remaining free pieces plus the start and end pieces of the partial solution sequence
together in left-right pairs such that no piece is assigned twice and the number of overlapping symbols between these remaining
pieces is maximized.

This method improves on a more basic heuristic in which conflicts in left-right assignments are ignored and all pieces are simply
assumed to match up with their preferred counterpart. The basic heuristic is faster, but also produces more nodes for the reel
search to go through until it finds the optimal solution.

With every step deeper into the estimate search tree, the algorithm resolves one more conflict between two pieces tied for the
same right-piece by re-assigning it to a different right-piece. This results in a breadth-first search with a constant branching
factor of 2.

Since the algorithm produces a continuous improvement over the basic heuristic, it can be interrupted at any time. It can e.g. be
limited to a maximum number of examined or memorized nodes.
'''
import logging
import copy
import functools
import ctypes

from heapq import heappush, heappop
from collections import namedtuple

# EstContext holds information about the estimate search environment that is relevant to the operations of EstNode
# as a flyweight type.
# overmat is the overlap matrix between the observed pieces.
# pref is a list that contains, for every observed piece, the ordered list of preferred right-hand pieces to assign.
# lefts is the set of free pieces plus Z of the partial solution.
# A is the left-hand end piece of the partial solution.
# Z is the right-hand end piece of the partial solution.
EstContext = namedtuple('EstContext', ['overmat', 'pref', 'lefts', 'A', 'Z'])

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
	def __init__(self, parent, cost, resolv=None, step=0):
		'''Initialize the EstNode.
		parent: the parent EstNode.
		resolv: an obs piece for which the assoc should be raised compared to the parent assoc to resolve a conflict.
		step: how much the assoc for the resolv piece should be raised compared to the parent.
		cost: the number of overlapping symbols that this assoc configuration is missing compared to the root assoc.
		'''
		# overmat, pref, lefts, _, _ = context
		# self.assoc = assoc
		self.parent = parent
		self.resolv = resolv
		self.step = step
		self.cost = cost

	def __lt__(self, other):
		'''Ordering for leaf heap.'''
		return self.cost < other.cost

	def __make_assoc(self, context):
		'''Generate this node’s assoc from the parent assoc + the resolv piece.
		Since the root node has no parent, it starts with the all-0 assoc. It will automatically convert to
		point only to valid right-hand pieces when run through __rhs during __find_conflict.
		'''
		try:
			self.assoc = copy.deepcopy(self.parent.assoc)
		except AttributeError:
			self.assoc = [-1 for _ in range(len(context.pref))]
			# update assoc to point to valid pieces
			for l in context.lefts:
				self.assoc[l] += self.__step(l, context)
			return

		self.assoc[self.resolv] += self.step
		# self.assoc[self.resolv] += 1


	# def __rhs(self, p, context):
	# 	'''Return the right-piece associated with the left-hand piece p according to this EstNode’s assoc.
	# 	self.assoc will be adjusted on demand to point to the next available right piece in the line.
	# 	Raise IndexError if valid right-hand pieces have been exhausted in this Node.
	# 	'''
	# 	_, pref, lefts, A, Z = context
	# 	pref_p = pref[p]   # preferred rhs for this piece
	# 	a = self.assoc[p]  # current preference cursor (index into pref_p) -> raise this until valid piece

	# 	while True:
	# 		rhs = pref_p[a] # Raise IndexError if valid right-hand pieces have been exhausted in this Node.
	# 		if (rhs == A) or ((rhs in lefts) and (rhs != Z)):
	# 			self.assoc[p] = a # save result of valid piece search
	# 			return rhs

	# 		a += 1 # try next preferred piece

	def __find_conflict(self, context):
		'''Return one conflict in self.assoc.

		A conflict is a left-hand to right-hand association such that
		pref[p][self.assoc[p]] == pref[q][self.assoc[q]], p != q, where p and q are both left-hand pieces
		(free pieces + Z).

		The representation of one such conflict, as returned by this method, is a tuple (p, q) of
		two of the left-hands (lhs) involved.

		If there are no more conflicts (meaning that the EstNode is a goal), returns nothing (None).
		If the assoc has already exhausted the pref list(s), raises IndexError.
		'''
		# Old docstring for __find_conflicts:
		'''Return every conflict in self.assoc.

		A conflict is a left-hand to right-hand association such that
		pref[p][self.assoc[p]] == pref[q][self.assoc[q]], p != q, where p and q are both left-hand pieces
		(free pieces + Z).

		The representation of one such conflict is a list of all the left-hands (lhs) involved.
		This method returns an iterable over all the detected conflicts.
		'''
		_, pref, lefts, _, _ = context
		# rhs_lhs = ((self.__rhs(p, context), p) for p in context.lefts)
		rhs_lhs = ((pref[p][self.assoc[p]], p) for p in context.lefts)
		rhs_lhs = sorted(rhs_lhs) # groupby requires sorted input

		rhs_prev = None
		for rhs_next, lhs_next in rhs_lhs:
			if rhs_prev == rhs_next:
				return lhs_prev, lhs_next # the two conflicted pieces fighting over rhs_next
			rhs_prev, lhs_prev = rhs_next, lhs_next

		# for rhs, pair_iter in itertools.groupby(rhs_lhs, key=itemgetter(0)):
		# 	lhs = list(map(itemgetter(1), pair_iter))
		# 	if len(lhs) > 1: # more than one lhs associated with the rhs is a conflict
		# 		yield lhs

	# def __resolutions(self, conflict):
	# 	'''Resolving a conflict involves nothing more than re-associating all but one of the involved
	# 	left-hand pieces. One resolution is a list of the pieces to be re-associated. For a conflict
	# 	with N involved pieces, there are thus N resolutions.
	# 	'''
	# 	for i in range(len(conflict)):
	# 		yield conflict[:i] + conflict[i+1:]

	def __step(self, resolv, context):
		'''Return the step that a successor node must add on top of the resolv pieces’s assoc for this node
		to arrive at a new valid assoc mapping (referring to one of the eligible pieces from the available
		right-hand pieces).
		Raise IndexError if valid right-hand pieces have been exhausted in this Node.
		'''
		_, pref, lefts, A, Z = context
		pref_p = pref[resolv]       # preferred rhs for this piece
		a = self.assoc[resolv] + 1  # current preference cursor (index into pref_p) -> raise this until valid piece

		while True:
			rhs = pref_p[a] # Raise IndexError if valid right-hand pieces have been exhausted in this Node.

			if (rhs == A) or ((rhs in lefts) and (rhs != Z)):
				return a - self.assoc[resolv]

			a += 1 # try next preferred piece

	def __resolv(self, resolv, context):
		'''Produce one successor to the current node by resolving a conflict involving the resolv piece.
		Raise IndexError if the conflict cannot be resolved by re-associating this piece (rhs have been exhausted).
		'''
		overmat, pref, _, _, _ = context
		assoc = self.assoc

		step = self.__step(resolv, context)
		rhs_before = pref[resolv][assoc[resolv]]
		rhs_after = pref[resolv][assoc[resolv] + step]
		cost = overmat[resolv][rhs_before] - overmat[resolv][rhs_after]
		return EstNode(self, cost, resolv, step)

	def expand(self, context):
		'''Evaluate the finer details of this node (conflicts) and test the goal condition.
		If this node is a goal, return False.
		If this node has conflicts and is thus not a goal, return True.
		'''
		self.__make_assoc(context)
		assoc = self.assoc

		try:
			p, q = self.__find_conflict(context)
			self.succ = list()

			try: self.succ.append(self.__resolv(p, context))
			except IndexError: pass

			try: self.succ.append(self.__resolv(q, context))
			except IndexError: pass

			return True

		except TypeError: # no conflict found (goal)
			return False

		# self.conflicts = list(self.__find_conflict(context))
		# return bool(self.conflicts)

	def successor(self, context):
		'''Return all successors to this EstNode in the estimate search tree.'''
		return self.succ

		# '''Generate all successors to this EstNode in the estimate search tree.'''
		# overmat, pref, _, _, _ = context
		# # conflicts = self.__find_conflicts(context)
		# conflicts = self.conflicts
		# resolutions = list(map(self.__resolutions, conflicts)) # for each conflict, we now have a list of possible resolutions for that conflict.

		# # Every successor node emerges from a plan.
		# # A plan is a collection of left-hand pieces that should be re-associated so that they might yield a conflict-free estimate.
		# # A plan is generated by attempting one (of several possible) resolutions for every conflict instance.
		# for plan in itertools.product(*resolutions):
		# 	succ_assoc = copy.deepcopy(self.assoc)
		# 	succ_cost = self.cost
		# 	valid = True

		# 	# advance is the piece index of one piece which we must re-associate by advancing its assoc entry to the next pref.
		# 	for advance in itertools.chain(*plan):
		# 		succ_assoc[advance] += 1
		# 		valid = valid and raise_assoc(succ_assoc, advance, context)
		# 		if not valid: break

		# 		# increase succ cost by # of overlap lost
		# 		pref_row = pref[advance]
		# 		self_pref = pref_row[self.assoc[advance]]
		# 		succ_pref = pref_row[succ_assoc[advance]]
		# 		self_overlap = overmat[advance][self_pref] 
		# 		succ_overlap = overmat[advance][succ_pref] 
		# 		loss = self_overlap - succ_overlap
		# 		assert loss >= 0
		# 		succ_cost += loss

		# 	if valid:
		# 		yield EstNode(succ_assoc, succ_cost, context)

class Heuristic:
	'''Represents the estimation algorithm and its environment (context).'''

	def __init__(self, overmat, pref, lobs):
		'''Initialize the heuristic with the global context.'''
		self.overmat = overmat
		self.pref = pref
		self.lobs = lobs

	def __calc_over(self, lefts, assoc):
		'''Get total overlap from the assoc configuration.'''
		over = 0

		for i in lefts:
			a = assoc[i]           # how many steps down in the preference this piece had to go
			p = self.pref[i][a]    # piece which is attached to i according to our configuration
			o = self.overmat[i][p] # overlapping symbols count
			over += o

		return over

	def __call__(self, lefts, A, Z):
		'''Run one estimation search for the given characteristics of the partial solution.'''
		overmat, pref, lobs = self.overmat, self.pref, self.lobs
		context = EstContext(overmat, pref, lefts, A, Z)

		root = EstNode(None, 0)
		leaves = [root] # bfs node heap
		clean_config = None # bfs goal
		over = 0

		resolv_steps = 100 # max iterations to try and resolve conflicts

		while leaves:
			node = heappop(leaves)

			search_more = node.expand(context)
			logging.debug('Est examine <<%s>>', node.assoc)	

			resolv_steps -= 1
			if resolv_steps <= 0:
				search_more = False # limit reached

			if not search_more: # this is goal
				clean_config = node.assoc
				over = self.__calc_over(lefts, node.assoc)
				break

			# logging.debug('Conflicts=%s', node.conflicts)

			for succ in node.successor(context):
				heappush(leaves, succ)

			# DEBUG: just one iteration	
			# over = calc_over(lefts, overmat, pref, node.assoc)
			# break

		H = overmat[Z][A]                                 # revert finished-loop assumption from cost g(n)
		H -= over

		lefts.remove(Z) # adopt lefts set for current node needs => is now the set of free pieces
		H += functools.reduce(lambda a, f: a + lobs[f], lefts, 0)   # total length of leftover pieces without overlap
		lefts.add(Z) # restore passed set

		H = max(0,H)
		logging.debug('H=%s, over=%s', H, over)
		return H

# DEBUG code
if __name__ == "__main__":
	import itertools
	import reels

	lestimate = ctypes.cdll.LoadLibrary("./estimate.so")
	create_context = lestimate.create_context # create_context function
	create_context.restype = ctypes.c_void_p
	estimate = lestimate.estimate # estimate function

	obs = ['abc', 'bcred', 'deab', 'de']
	overmat, elim = reels.make_overmat(obs)
	free = set([i for i in range(len(obs)) if i not in elim])        # complete overlap to reduce search space
	lobs = [len(o) for o in obs]
	pref = reels.make_pref(overmat, free)
	# reel_context = reels.ReelContext(obs, lobs, free, overmat, pref)
	lefts = [0, 1, 2]  # in real code, convert lefts set to list
	a = 0
	z = 0

	n = len(obs)                 # length of obs list, overmat etc.
	p = len(obs) - len(elim) - 1 # length of one pref list
	l = len(lefts)

	# adapt lists to uniform lengths and types for passing to C interface
	flat_overmat = [v if v != None else -1 for v in itertools.chain(*overmat)]
	padded_pref = [pr if pr != [] else [-1 for j in range(p)] for pr in pref] # pref with uniform lengths of p for member arrays
	flat_pref = list(itertools.chain(*padded_pref))

	OvermatArray = ctypes.c_int * (n * n)  # C array type to store overmat
	PrefArray = ctypes.c_int * (n * p)     # C array type to store pref
	LobsArray = ctypes.c_int * n           # C array type to store lobs
	LeftsArray = ctypes.c_int * l          # C array type to store lefts
	
	print("overmat = ", overmat)
	print("flat overmat = ", flat_overmat)
	print("pref = ", pref)
	print("flat pref = ", flat_pref)

	c_overmat = OvermatArray(*flat_overmat)
	c_pref = PrefArray(*flat_pref)
	c_lobs = LobsArray(*lobs)
	c_lefts = LeftsArray(*lefts)

	context = create_context(n, c_overmat, p, c_pref, c_lobs)
	print("context = {0}".format(hex(context)))
	result = estimate(l, c_lefts, a, z, context)
	lestimate.destroy_context(context)

	print("lestimate.estimate({0},{1},{2},{3},{4}) = {5}".format(l, lefts, a, z, hex(context), result))
	# print("lestimate.EstContext = ", lestimate.EstContext)
