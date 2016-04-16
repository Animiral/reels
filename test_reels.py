#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

# Some tests for the reels.py module

import reels
import logging
import sys

def test():
	logging.info('TEST...')

	test_overlap()
	test_solution()
	test_est()
	test_purify()
	test_successor()

	logging.info('TEST DONE')

def test_overlap():
	sys.stderr.write('Test overlap(left,right):')

	cases = [
		('abc','defg',0),
		('abc','cde',1),
		('abcdef','defe',3),
		('abcdef','cde',0)
	]

	for c in cases:
		left, right, expected = c
		sys.stderr.write('\toverlap({0},{1}) == {2}: '.format(left,right,expected))
		actual = reels.overlap(left, right)
		if actual == expected: 
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_solution():
	sys.stderr.write('Test solution(sequence):\n')

	reels.g_obs = ['abc','cde','cab']
	reels.g_overlap = [[0, 1, 1], [0, 0, 0], [2, 0, 0]]
	cases = [
		([0,1],'abcde'),
		([1,0],'cdeabc'),
		([0,2],'abcab')
	]

	for c in cases:
		sequence, expected = c
		sys.stderr.write('\tsolution({0}) == {1}: '.format(sequence,expected))
		actual = reels.solution(sequence)
		if actual == expected: 
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_est():
	sys.stderr.write('Test est(node):\n')

	reels.g_obs = ['abc','cdef']
	reels.g_overlap = [[0,0],[1,0]]
	cases = [
		(reels.ReelNode([0],[1],3),3+3),
		(reels.ReelNode([1],[0],4),4+2),
		(reels.ReelNode([0,1],[],6),6)
	]

	for c in cases:
		node, expected = c
		sys.stderr.write('\test({0}) == {1}: '.format(node,expected))
		actual = reels.est(node)
		if actual == expected:
			sys.stderr.write('OK\n')
		else:                  
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

	reels.g_obs = ['318', '931', '8079553b00a', '180', '0ab93']
	reels.g_overlap = [[0, 0, 1, 2, 0], [2, 0, 0, 1, 0], [0, 0, 0, 0, 2], [0, 0, 2, 0, 1], [1, 2, 0, 0, 0]]
	cases = [
		(reels.ReelNode([2],[0,1,3,4],11),11+1),
		(reels.ReelNode([0,2],[1,4],13),13+1),
		(reels.ReelNode([1,0,2],[4],14),14+1)
	]

	for c in cases:
		node, expected = c
		sys.stderr.write('\test({0}) == {1}: '.format(node,expected))
		actual = reels.est(node)
		if actual == expected:
			sys.stderr.write('OK\n')
		else:
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

def test_purify():
	sys.stderr.write('Test purify(node):\n')

	def _run():
		for c in cases:
			node, expected = c
			sys.stderr.write('\tpurify({0}) == {1}: '.format(node,expected))
			actual = reels.purify(node)
			if actual == expected: 
				sys.stderr.write('OK\n')
			else:                  
				sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(expected,actual))

	reels.g_obs = ['abc','cde','bcd','ac']
	reels.g_overlap = [[0,1,2,0],[0,0,0,0],[0,2,0,0],[0,1,0,0]]
	cases = [
		(reels.ReelNode([0,1],[2,3],5),reels.ReelNode([0,1],[3],5)),
		(reels.ReelNode([2,0],[1,3],6),reels.ReelNode([2,0],[1,3],6))
	]

	_run()

	reels.g_obs=['AAA', 'AAB', 'ABBCA', 'CAAB', 'BB']
	reels.g_overlap=[[0, 2, 1, 0, 0], [0, 0, 2, 0, 1], [1, 1, 0, 2, 0], [0, 3, 2, 0, 1], [0, 0, 0, 0, 0]]
	cases = [
		(reels.ReelNode([2],[0,1,2,3,4],5),reels.ReelNode([2],[0,1,3],5))
	]

	_run()

def test_successor():
	sys.stderr.write('Test successor(node):\n')

	reels.g_obs = ['318', '931', '8079553b00a', '180', '0ab93']
	reels.g_overlap = [[0, 0, 1, 2, 0], [2, 0, 0, 1, 0], [0, 0, 0, 0, 2], [0, 0, 2, 0, 1], [1, 2, 0, 0, 0]]

	n0 = reels.ReelNode([2],[0,1,3,4],11)

	for S in reels.successor(n0):
		est1 = S.est
		est2 = reels.est(S)

		sys.stderr.write('\tsuccessor(n0):est({0}) == {1}: '.format(est1,est2))
		if S.est == est2:
			sys.stderr.write('OK\n')
		else:
			sys.stderr.write('FAIL (expected={0}, actual={1})\n'.format(est2,S.est))

if __name__ == "__main__":
	test()
