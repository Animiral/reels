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

	parser = argparse.ArgumentParser(description=g_desc)
	parser.add_argument('files', metavar='FILE', nargs='*', help='input file(s)')
	parser.add_argument('-o', '--out_file', required=True, help='reels output file (last line contains solution)')
	parser.add_argument('-s', '--solution', help='reference solution string - if provided, length and loop-invariant match is checked')

	args = parser.parse_args()
	return args.files, args.out_file, args.solution

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
	files, out_file, reference = handle_args()
	obs = reels.read_obs(files)
	sol = read_solution(out_file)

	valid = True

	for O in obs:
		if not O in (sol*2):
			sys.stdout.write('Piece missing in solution: "{0}"\n'.format(O))
			valid = False

	if reference:
		len_sol = len(sol)
		len_ref = len(reference)

		if len_sol > len_ref:
			sys.stdout.write('Solution is too long. ({0} chars; {1} chars reference)\n'.format(len_sol, len_ref))
			valid = False
		else:
			if not (reference in (sol*2) and sol in (reference*2)):
				sys.stdout.write('Solution does not match reference. (output="{0}"; reference="{1}")\n'.format(sol, reference))
				# Solution still passes if the reference happens not to be the only or optimal solution

	else:
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
