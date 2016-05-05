#!/usr/bin/env pypy3
# -*- coding: UTF-8 -*-

'''This is a helper script to batch-process inputs to reels.py.
It reads all input files from a directory and processes them one by one using the reels module.
The outputs of an invocation of reels.py on file.ext are written to <prefix>file.ext, where
prefix is either specified as a parameter or 's_' by default.
'''

import reels
import logging

def dofile(workdir, in_file, args, reels_args):
	'''Process one input file.'''
	import functools
	import io
	from os.path import join

	out_file = args.prefix + in_file
	logging.info('%s -> %s', in_file, out_file)
	in_file = join(workdir, in_file)
	out_file = join(workdir, out_file)

	csv = reels_args.csv or in_file.endswith('.csv') # special case: if file ext indicates CSV, always parse CSV
	read_obs_func = functools.partial(reels.read_obs_csv, dialect=reels_args.dialect) if csv else reels.read_obs
	free, context = reels.setup(in_file, read_obs_func)
	search = {'astar': reels.astar, 'dfs': reels.dfs} [reels_args.algorithm]
	out_fd = io.open(out_file, 'a' if args.keep else 'w')
	format_solution = (lambda s: ','.join(s)) if csv else (lambda s: ''.join(s))

	reels.run(free, context, search, reels_args.limit, reels_args.full, reels_args.solutions, out_fd, format_solution)

def handle_args():
	'''Parse and handle command arguments.'''
	import argparse

	parser = argparse.ArgumentParser(description='''
		Reads input files from DIRECTORY and writes the solutions to output files with the specified prefix.
		If no directory is specified, reads from the current working directory.
		If no prefix is specified, the default is 's_'.
		All input files that match the choose regular expression are considered for processing.
		Files starting with the prefix, ending with .py or matching the ignore regular expression are not processed.
		If there is no choose regular expression, '.*' is the default.
		If there is no ignore regular expression, no files are excluded by it.
		Arguments which are not recognized by batch_reels.py are passed to reels.py,
		except for in_file and out_file, which are ignored.
		''')
	parser.add_argument('-d', '--directory', default='.', help='directory which contains the input files')
	parser.add_argument('-k', '--keep', action='store_true', default=False, help='do not truncate solution files')
	parser.add_argument('-p', '--prefix', type=str, default='s_', help='prefix string for solution files')
	parser.add_argument('-i', '--ignore', type=str, help='regular expression of file names to skip')
	parser.add_argument('-c', '--choose', type=str, default='.*', help='regular expression of file names to process')

	args, more = parser.parse_known_args()
	reels_args = reels.handle_args(more)
	return args, reels_args

def main():
	import re
	from os import walk

	args, reels_args = handle_args()

	workdir = args.directory
	for (_, _, filenames) in walk(workdir):
	    for in_file in filenames:
	    	skip = not re.search(args.choose, in_file)
	    	skip = skip or in_file.startswith(args.prefix)  # known solution file => this is not an input
	    	skip = skip or in_file.endswith('.py')          # python scripts - frequently occur in working directory
	    	skip = skip or (args.ignore and re.search(args.ignore, in_file))  # explicitly specified ignore files
	    	if not skip:
		    	dofile(workdir, in_file, args, reels_args)
	    break

if __name__ == '__main__':
	main()
