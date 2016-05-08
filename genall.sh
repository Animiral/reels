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

mkdir -p in
rm in/*


for i in $(seq 3)
do
	for k in 10 20 30 40 45 50 55 60
	do
		./genreel.py --method=random 12 $((3*k)) $k      | tee >(head -n 1 > in/s_r$k_$i.txt) | tail -n $k > in/r$k_$i.in
		./genreel.py --method=min_overlap 12 $((3*k)) $k | tee >(head -n 1 > in/s_p$k_$i.txt) | tail -n $k > in/p$k_$i.in
		./genreel.py --method=max_overlap 12 $((3*k)) $k | tee >(head -n 1 > in/s_q$k_$i.txt) | tail -n $k > in/q$k_$i.in
	done
done

# CSV test inputs
./genreel.py --csv --method=random 12 30 10 | tee >(head -n 1 > in/s_csv10.txt)   | tail -n 10  > in/csv10.csv
./genreel.py --csv --method=random 12 60 20 | tee >(head -n 1 > in/s_csv20.txt)   | tail -n 20  > in/csv20.csv
./genreel.py --csv --method=random 12 90 30 | tee >(head -n 1 > in/s_csv30.txt)   | tail -n 30  > in/csv30.csv
./genreel.py --csv --method=random 12 120 40 | tee >(head -n 1 > in/s_csv40.txt)   | tail -n 40  > in/csv40.csv
./genreel.py --csv --method=random 12 150 50 | tee >(head -n 1 > in/s_csv50.txt)   | tail -n 50  > in/csv50.csv
