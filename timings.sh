#!/bin/bash
# Timings of reels.py

# p10.in : 0.036
# p15.in : 9.448
# p5.in : 0.000
# q10.in : 0.038
# q15.in : 0.080
# q5.in : 0.000
# r10.in : 0.022
# r15.in : 1.029
# r5.in : 0.001

for f in in/*.in
do
	echo $(basename $f) : $(./profile_reels.py ts "$f")
done
