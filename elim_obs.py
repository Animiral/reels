#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

'''
Read reels.py input sets from the files specified in the command line and eliminate from them all initially redundant pieces.
reels.py will do that by itself before running the search, but this script can shorten the test cases to get their actual difficulty into the profile for better statistics.
'''

import sys
import io
import reels

def main():
	if len(sys.argv) <= 1:
		sys.stderr.write('usage: {0} FILE(S)...\n\nMinimizes the reels.py problems contained in the files. CSV is auto-detected by file extension.\n')
		return

	for in_file in sys.argv[1:]:
		try:
			csv_format = in_file.endswith('.csv')
			read_obs_func = reels.read_obs_csv if csv_format else reels.read_obs
			obs = read_obs_func(in_file)
			_, elim = reels.make_overmat(obs)
			short_obs = [o for i, o in enumerate(obs) if i not in elim] # eliminate pieces with complete overlap

			with io.open(in_file,'w') as out_fd:
				for o in short_obs:
					obs_str = ','.join(o) if csv_format else ''.join(o)
					out_fd.write(obs_str + '\n')

			sys.stdout.write('{0}: eliminated dupes and {1} pieces -> n={2}\n'.format(in_file, len(elim), len(short_obs)))
		except IOError as e:
			sys.stderr.write('{0}\n'.format(e))

if __name__ == "__main__":
	main()
