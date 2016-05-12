#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

'''Some tests for the reels.py module.'''

import logging
import sys

from reels import ReelContext, ReelNode, overlap, make_overmat, make_pref
Node = ReelNode

def test():
	if sys.argv[-1] == '--debug': logging.basicConfig(level=logging.DEBUG)
	else:                         logging.basicConfig(level=logging.WARNING)

	logging.info('TEST...')

	test_overlap()
	test_pref()
	test_solution()
	test_est()
	test_successor()

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
	pref = make_pref(obs, overmat, [0,1,2])
	context = ReelContext(obs, overmat, pref)
	cases = [
		(Node([0,1],[2],0,context),'abcde'),
		(Node([1,0],[2],0,context),'cdeabc'),
		(Node([0,2],[1],0,context),'abcab')
	]

	for c in cases:
		node, expected = c
		sys.stderr.write('\t__solution({0}) == {1}: '.format(node.sequence,expected))
		actual = node._ReelNode__solution(context)
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
		actual = make_pref(obs, overmat, free)

		if actual == expected: 
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_est():
	sys.stderr.write('Test est(node):\n')

	obs1 = ['abc','cdef']
	obs2 = ['318', '931', '8079553b00a', '180', '0ab93']
	obs3 = ['babb','bcb','bba']   # babbcb vs. babbabcb

	cases = [
		(obs1,[0],[1],3,        3+3),
		(obs1,[1],[0],4,        4+2),
		(obs1,[0,1],[],6,       6),
		(obs2,[2],[0,1,3,4],11, 11+4),
		(obs2,[0,2],[1,4],13,   13+2),
		(obs2,[1,0,2],[4],14,   14+1),
		(obs3,[0],[1,2],3,      6)
	]

	for c in cases:
		obs, seq, free, cost, expected = c
		overmat, _ = make_overmat(obs)
		pref = make_pref(obs, overmat, list(range(len(obs))))
		context = ReelContext(obs, overmat, pref)
		node = Node(seq,free,cost,context)

		sys.stderr.write('\t{0}.__calc_est() == {1}: '.format(node,expected))
		actual = node._ReelNode__calc_est(context)
		if actual == expected:
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_successor():
	sys.stderr.write('Test successor(node):\n')

	obs = ['318', '931', '8079553b00a', '180', '0ab93']
	overmat, _ = make_overmat(obs)
	pref = make_pref(obs, overmat, [0,1,2,3,4])
	context = ReelContext(obs, overmat, pref)

	n0 = Node([2],[0,1,3,4],11,context)

	for S in n0.successor(context):
		est1 = S.est
		est2 = S._ReelNode__calc_est(context)

		sys.stderr.write('\tsuccessor(n0):est({0}) == {1}: '.format(est1,est2))
		if S.est == est2:
			sys.stderr.write('OK\n')
		else:
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(est2,S.est))

if __name__ == "__main__":
	test()
