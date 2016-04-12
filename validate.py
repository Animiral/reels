#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

g_desc = '''
This is a validator for reels.py. It does nothing more than check the output of reels.py for two things:
 1) Every input piece must occur in the output string.
 2) The output string must not be longer than the sum of input lengths.
'''

# usage:

import sys
import io
import argparse
import reels

# Parse and handle command arguments
def handle_args():
	global g_desc
	global g_files
	global g_out_file

	parser = argparse.ArgumentParser(description=g_desc)
	parser.add_argument('files', metavar='FILE', nargs='*', help='input file(s)')
	parser.add_argument('-o', '--out_file', dest='out_file', required=True, help='reels output file (last line contains solution)')

	args = parser.parse_args()
	g_files = args.files
	g_out_file = args.out_file

# Reads reels.py output solution from out_file.
# We assume that the last non-empty line in the output file is the solution that we want to check against.
def read_solution(out_file):
	f = io.open(out_file,'r')
	lines = f.read().splitlines()
	for L in reversed(lines):
		L = L.strip()
		if len(L) > 0:
			return L
	else:
		raise RuntimeError('No solution found in %s.', out_file)

def main():
	global g_files
	global g_out_file

	handle_args()
	obs = reels.read_obs(g_files)
	sol = read_solution(g_out_file)

	valid = True

	for O in obs:
		if not O in sol:
			sys.stdout.write('Piece missing in solution: "{0}"\n'.format(O))
			valid = False

	len_sol = len(sol)
	len_obs = len(''.join(obs))

	if len_obs < len_sol:
		sys.stdout.write('Solution is too long. ({0} chars; {1} chars observed)\n'.format(len_sol, len_obs))
		valid = False

	if valid:
		sys.stdout.write('PASSED\n')
	else:
		sys.stdout.write('FAILED\n')

if __name__ == "__main__":
	main()
