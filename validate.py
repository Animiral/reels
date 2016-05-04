#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

'''
This is a validator for reels.py. It does nothing more than check the output of reels.py for two things:
 1) Every input piece must occur in the output string.
 2) The output string must not be longer than the sum of input lengths.
'''

import sys
import io
import argparse
import reels

def handle_args():
	'''Parse and handle command arguments.'''
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('file', metavar='FILE', type=str, nargs='?', help='input file')
	parser.add_argument('-o', '--out_file', required=True, help='reels output file (last line contains solution)')
	parser.add_argument('-s', '--solution', help='reference solution string - if provided, length and loop-invariant match is checked')
	parser.add_argument('--csv', action='store_true', default=False, help='specify default input format as CSV')

	a = parser.parse_args()

	if 'csv' in a.file: # special case: if file ext indicates CSV, always parse CSV
		a.csv = True

	return a.file, a.out_file, a.solution, a.csv

def read_solution(out_file):
	'''Read reels.py output solution from out_file.
	We assume that the last non-empty line in the output file is the solution that we want to check against.
	'''
	f = io.open(out_file,'r')
	lines = f.read().splitlines()
	for L in reversed(lines):
		L = L.strip()
		if len(L) > 0:
			return L
	else:
		raise RuntimeError('No solution found in %s.', out_file)

def main():
	import functools

	file, out_file, reference, csv = handle_args()

	if csv:
		read_obs_func = functools.partial(reels.read_obs_csv, dialect=None)
		conv_line = lambda line: line.split(',')
		reference = reference.split(',')
	else:
		read_obs_func = reels.read_obs
		conv_line = lambda line: list(line)
		reference = list(reference)

	obs = read_obs_func(file)
	sol = conv_line(read_solution(out_file))

	valid = True
	alt_solution = False    # turns to True if the output solution is valid, but different from the reference solution

	for O in obs:
		if not reels.is_subsequence((sol*2), O):
			sys.stdout.write('Piece missing in solution: "{0}"\n'.format(O))
			valid = False

	if reference:
		len_sol = len(sol)
		len_ref = len(reference)

		if len_sol > len_ref:
			sys.stdout.write('Solution is too long. ({0} symbols; {1} symbols reference)\n'.format(len_sol, len_ref))
			valid = False
		else:
			if not (reels.is_subsequence((sol*2), reference) and reels.is_subsequence((reference*2), sol)):
				# Solution still passes if the reference happens not to be the only or optimal solution
				alt_solution = True

	else:
		len_sol = len(sol)
		len_obs = len(''.join(obs))

		if len_obs < len_sol:
			sys.stdout.write('Solution is too long. ({0} symbols; {1} symbols observed)\n'.format(len_sol, len_obs))
			valid = False

	if valid:
		if alt_solution:
			# sys.stdout.write('Solution does not match reference. (output="{0}"; reference="{1}")\n'.format(sol, reference))
			sys.stdout.write('PASSED*\n')
		else:
			sys.stdout.write('PASSED\n')
	else:
		sys.stdout.write('FAILED\n')

if __name__ == "__main__":
	main()
