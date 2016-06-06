#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

'''Some tests for the reels.py module.'''

import logging
import sys

from reels import ReelContext, ReelNode, EstNode, EstContext, overlap, make_overmat, make_pref
Node = ReelNode

def test():
	if sys.argv[-1] == '--debug': logging.basicConfig(level=logging.DEBUG)
	else:                         logging.basicConfig(level=logging.WARNING)

	logging.info('TEST...')

	test_overlap()
	test_pref()
	test_solution()
	test_est()
	test_reel_successor()
	test_est_successor()

	logging.info('TEST DONE')

def test_overlap():
	sys.stderr.write('Test overlap(left,right):\n')

	cases = [
		('abc','defg',0),
		('abc','cde',1),
		('abcdef','defe',3),
		('abcdef','cde',0)
	]

	for c in cases:
		left, right, expected = c
		sys.stderr.write('\toverlap({0},{1}) == {2}: '.format(left,right,expected))
		actual = overlap(left, right)
		if actual == expected: 
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_solution():
	sys.stderr.write('Test node.__solution():\n')

	obs = ['abc','cde','cab']
	overmat, _ = make_overmat(obs)
	free = list(range(len(obs)))
	lobs = [len(o) for o in obs]
	pref = make_pref(overmat, free)
	context = ReelContext(obs, lobs, free, overmat, pref)

	# cases = [
	# 	(Node([0,1],[2],0,context),'abcde'),
	# 	(Node([1,0],[2],0,context),'cdeabc'),
	# 	(Node([0,2],[1],0,context),'abcab')
	# ]

	root0 = Node(parent=None, piece=0, cost=3, context=context)
	root1 = Node(parent=None, piece=1, cost=3, context=context)

	for n in [root0, root1]:
		n.expand(context)

	node01 = Node(parent=root0, piece=1, cost=5, context=context)
	node02 = Node(parent=root0, piece=2, cost=5, context=context)
	node10 = Node(parent=root1, piece=0, cost=6, context=context)

	for n in [node01, node02, node10]:
		n.expand(context)

	cases = [
		(node01,'abcde'),
		(node10,'cdeabc'),
		(node02,'abcab')
	]

	for c in cases:
		node, expected = c
		sys.stderr.write('\t__solution({0}) == {1}: '.format(node._ReelNode__sequence(), expected))
		# actual = node._ReelNode__solution(context)
		actual = node._ReelNode__solution(node._ReelNode__sequence(), context)
		if actual == expected: 
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_pref():
	sys.stderr.write('Test make_pref():\n')

	cases = [
		(['abcde','eab','dea'], [[2,1],[0,2],[1,0]]),
		(['318', '931', '8079553b00a', '180', '0ab93'], [[3,2,1,4],[0,3,2,4],[4,0,1,3],[2,4,0,1],[1,0,2,3]])
	]

	for c in cases:
		obs, expected = c
		overmat, _ = make_overmat(obs)
		# overmat = [[0, 1, 2], [2, 0, 0], [1, 2, 0]]
		free = list(range(len(obs))) # there are no redundant pieces in cases

		sys.stderr.write('\tmake_pref({0},<mat>,<free>) == {1}: '.format(obs, expected))
		actual = make_pref(overmat, free)

		if actual == expected: 
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_est():
	sys.stderr.write('Test est(node):\n')

	obs1 = ['abc','cdef']
	obs2 = ['318', '931', '8079553b00a', '180', '0ab93']
	obs3 = ['babb','bcb','bba']   # babbcb vs. babbabcb
	lobs1 = [len(o) for o in obs1]
	lobs2 = [len(o) for o in obs2]
	lobs3 = [len(o) for o in obs3]

	overmat1, _ = make_overmat(obs1)
	pref1 = make_pref(overmat1, list(range(len(obs1))))
	free1 = list(range(len(obs1)))
	context1 = ReelContext(obs1, lobs1, free1, overmat1, pref1)

	overmat2, _ = make_overmat(obs2)
	pref2 = make_pref(overmat2, list(range(len(obs2))))
	free2 = list(range(len(obs2)))
	context2 = ReelContext(obs2, lobs2, free2, overmat2, pref2)

	overmat3, _ = make_overmat(obs3)
	pref3 = make_pref(overmat3, list(range(len(obs3))))
	free3 = list(range(len(obs3)))
	context3 = ReelContext(obs3, lobs3, free3, overmat3, pref3)

	obs1_node0 = Node(parent=None, piece=0, cost=3, context=context1)
	obs1_node0.expand(context1)
	obs1_node1 = Node(parent=None, piece=1, cost=4, context=context1)
	obs1_node01 = Node(parent=obs1_node0, piece=1, cost=6, context=context1)
	obs2_node0 = Node(parent=None, piece=0, cost=3, context=context2)
	obs2_node0.expand(context2)
	obs2_node02 = Node(parent=obs2_node0, piece=2, cost=13, context=context2)
	obs2_node1 = Node(parent=None, piece=1, cost=3, context=context2)
	obs2_node1.expand(context2)
	obs2_node10 = Node(parent=obs2_node1, piece=0, cost=4, context=context2)
	obs2_node10.expand(context2)
	obs2_node102 = Node(parent=obs2_node10, piece=2, cost=14, context=context2)
	obs2_node2 = Node(parent=None, piece=2, cost=11, context=context2)
	obs3_node0 = Node(parent=None, piece=0, cost=3, context=context3)

	cases = [
		(obs1_node0, context1,   3+3),
		(obs1_node1, context1,   4+2),
		(obs1_node01, context1,  6),
		(obs2_node2, context2,   11+4),
		(obs2_node02, context2,  13+6-2),
		(obs2_node102, context2, 14+6-2),
		(obs3_node0, context3,   3+1+3-2)
		# (obs1,[0],[1],3,        3+3),
		# (obs1,[1],[0],4,        4+2),
		# (obs1,[0,1],[],6,       6),
		# (obs2,[2],[0,1,3,4],11, 11+4),
		# (obs2,[0,2],[1,4],13,   13+2),
		# (obs2,[1,0,2],[4],14,   14+1),
		# (obs3,[0],[1,2],3,      6)
	]

	for c in cases:
		node, context, expected = c

		sys.stderr.write('\t{0}.__calc_est() == {1}: '.format(node,expected))
		actual = node._ReelNode__calc_est(context)
		if actual == expected:
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_reel_successor():
	sys.stderr.write('Test ReelNode.successor():\n')

	obs = ['318', '931', '8079553b00a', '180', '0ab93']
	overmat, _ = make_overmat(obs)
	free = list(range(len(obs)))
	lobs = [len(o) for o in obs]
	pref = make_pref(overmat, free)
	context = ReelContext(obs, lobs, free, overmat, pref)

	# n0 = Node([2],[0,1,3,4],11,context)
	n0 = Node(parent=None, piece=2, cost=11, context=context)
	n0.expand(context)

	for S in n0.successor(context):
		est1 = S.est
		est2 = S._ReelNode__calc_est(context)

		sys.stderr.write('\tsuccessor(n0):est({0}) == {1}: '.format(est1,est2))
		if S.est == est2:
			sys.stderr.write('OK\n')
		else:
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(est2,S.est))

def test_est_successor():
	sys.stderr.write('Test EstNode.successor():\n')

	obs = ['319', '931', '0ab93']
	overmat, _ = make_overmat(obs)
	pref = make_pref(overmat, list(range(len(obs))))  # [[1,2], [0,2], [1,0]]
	lefts = [0,1,2]
	A = 1
	Z = 1
	context = EstContext(overmat, pref, lefts, A, Z)

	n0 = EstNode(None, 0)

	n0.expand(context)
	for S in n0.successor(context):
		S.expand(context)
		assoc_sum = S.assoc[0] + S.assoc[2]
		expected = 1

		sys.stderr.write('\tsuccessor(n0):assoc_sum({0}) == {1}: '.format(S.assoc, expected))
		if assoc_sum == expected:
			sys.stderr.write('OK\n')
		else:
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected, assoc_sum))

if __name__ == "__main__":
	test()
