#!/bin/bash
# This script generates a standard suite of test cases for reels.py using genreel.py.

# Generated problem files are written to the in/ directory and named “xN”, where x is one of:
#   - r for random observations
#   - p for min_overlap
#   - q for max_overlap
# and N is the number of observations in the input.
# Solution files are named “s_xN” and correspond to one problem file each.
# All other contents of the in/ directory are deleted.

# NOTE: reel size is usually 3x obs count, except for max_overlap, where the reel size must be equal to the obs count.

# problem | sym count | reel size | obs count
# ===========================================
#      r5 |        12 |        15 |         5
#     r10 |        12 |        30 |        10
#     r15 |        12 |        45 |        15
#     r20 |        12 |        60 |        20
#     p5  |        12 |        15 |         5
#     p10 |        12 |        30 |        10
#     p15 |        12 |        45 |        15
#     p20 |        12 |        60 |        20
#     q5  |        12 |         5 |         5
#     q10 |        12 |        10 |        10
#     q15 |        12 |        15 |        15
#     q20 |        12 |        20 |        20

mkdir -p in
rm in/*

./genreel.py --method=random 12 15 5  | tee >(head -n 1 > in/s_r5.txt)    | tail -n 5   > in/r5.in
./genreel.py --method=random 12 30 10 | tee >(head -n 1 > in/s_r10.txt)   | tail -n 10  > in/r10.in
./genreel.py --method=random 12 45 15 | tee >(head -n 1 > in/s_r15.txt)   | tail -n 15  > in/r15.in
# ./genreel.py --method=random 12 60 20 | tee >(head -n 1 > in/s_r20.txt)   | tail -n 20  > in/r20.in

./genreel.py --method=min_overlap 12 15 5  | tee >(head -n 1 > in/s_p5.txt)    | tail -n 5   > in/p5.in
./genreel.py --method=min_overlap 12 30 10 | tee >(head -n 1 > in/s_p10.txt)   | tail -n 10  > in/p10.in
./genreel.py --method=min_overlap 12 45 15 | tee >(head -n 1 > in/s_p15.txt)   | tail -n 15  > in/p15.in
# ./genreel.py --method=min_overlap 12 60 20 | tee >(head -n 1 > in/s_p20.txt)   | tail -n 20  > in/p20.in

./genreel.py --method=max_overlap 12 15 5  | tee >(head -n 1 > in/s_q5.txt)    | tail -n 5   > in/q5.in
./genreel.py --method=max_overlap 12 30 10 | tee >(head -n 1 > in/s_q10.txt)   | tail -n 10  > in/q10.in
./genreel.py --method=max_overlap 12 45 15 | tee >(head -n 1 > in/s_q15.txt)   | tail -n 15  > in/q15.in
# ./genreel.py --method=max_overlap 12 60 20 | tee >(head -n 1 > in/s_q20.txt)   | tail -n 20  > in/q20.in