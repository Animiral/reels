#!/usr/bin/env python
# random generator for reels.py problems
# usage: genreel.py <reel size> <obs count> <obs min size> <obs max size>

import argparse
import random

def make_pool(S):
	if S > 36: raise RuntimeError('More than 36 symbols not implemented')

	for i in range(0,10):
		if i < S: yield str(i)

	for i in range(0,26):
		if i < S-10: yield chr(97+i)

def sample_pool(P,N):
	for i in range(0,N):
		yield random.choice(P)

def make_reel(N,S):
	pool = list(make_pool(S))
	R = sample_pool(pool,N)
	return ''.join(R)

# Old implementation: simple random substrings
# def observations(reel,N,S,T):
# 	for i in range(0,N):
# 		L = random.randint(S,T)
# 		k = random.randint(0,len(reel)-L)
# 		yield reel[k:k+L]

# Updated implementation: cut the reel into segments at random cut points, each of which becomes a piece, with a predefined overlap of 2
def observations(reel,N):
	R = len(reel)
	cuts = random.sample(range(0,R), N)
	cuts.sort()

	for i in range(0,N):
		low = cuts[i] -1
		high = cuts[(i+1)%N] +1

		if low < 0: low += R
		if high < low: high += R

		yield (reel*2)[low:high]

def handle_args():
	parser = argparse.ArgumentParser(description='Random generator for reels.py problems')
	parser.add_argument('reel_size', type=int, help='number of symbols in the generated reel')
	parser.add_argument('sym_count', type=int, help='number of distinct symbols in the alphabet')
	parser.add_argument('obs_count', type=int, help='number of observations to generate')
	# NOTE: observation size is now determined by the random size of the piece
	# parser.add_argument('obs_min', type=int, help='minimum size of an observation')
	# parser.add_argument('obs_max', type=int, help='maximum size of an observation')
	args = parser.parse_args()
	return args.reel_size, args.sym_count, args.obs_count # , args.obs_min, args.obs_max

def random_observations(reel_size, sym_count, obs_count): # , obs_min, obs_max):
	random.seed()
	reel = make_reel(reel_size, sym_count)
	obs = list(observations(reel, obs_count)) #, obs_min, obs_max))
	random.shuffle(obs)      # input order should not matter, but shuffle just to be safe
	return (reel, obs)

# start the bus
def main():
	args = handle_args()
	reel, obs = random_observations(*args)

	print(reel)
	for o in obs:
		print(o)

if __name__ == '__main__':
	main()
