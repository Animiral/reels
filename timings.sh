#!/bin/bash
# Timings of reels.py

# reel size | sym count | obs count | time(s)
# ===========================================
#        15 |        12 |         5 |   0.001
#        45 |        12 |        15 |   0.023
#        75 |        12 |        25 |   0.165
#       115 |        12 |        35 |   
#       150 |        12 |        50 | 
#       225 |        12 |        75 | 
#       300 |        12 |       100 | 
#       375 |        12 |       125 | 
#       450 |        12 |       150 | 

mkdir -p in
rm in/*

./genreel.py 15 12 5    | tee >(head -n 1 > in/solution5.txt)   | tail -n 5   > in/problem5.in
./genreel.py 45 12 15   | tee >(head -n 1 > in/solution15.txt)  | tail -n 15  > in/problem15.in
./genreel.py 75 12 25   | tee >(head -n 1 > in/solution25.txt)  | tail -n 25  > in/problem25.in
./genreel.py 115 12 35  | tee >(head -n 1 > in/solution35.txt)  | tail -n 35  > in/problem35.in
./genreel.py 150 12 50  | tee >(head -n 1 > in/solution50.txt)  | tail -n 50  > in/problem50.in
./genreel.py 225 12 75  | tee >(head -n 1 > in/solution75.txt)  | tail -n 75  > in/problem75.in

echo "problem5: " $(./profile_reels.py ts in/problem5.in )
echo "problem15: " $(./profile_reels.py ts in/problem15.in )
echo "problem25: " $(./profile_reels.py ts in/problem25.in )
echo "problem35: " $(./profile_reels.py ts in/problem35.in )

# old genreel
# ./genreel.py 15 12 5 3 8     | tee >(head -n 1 > solutions.txt) | tail -n 5 > problem5.in
# ./genreel.py 20 12 10 3 8    | tee >(head -n 1 >> solutions.txt) | tail -n 10 > problem10.in
# ./genreel.py 50 12 25 3 8    | tee >(head -n 1 >> solutions.txt) | tail -n 25 > problem25.in
# ./genreel.py 75 12 50 3 8    | tee >(head -n 1 >> solutions.txt) | tail -n 50 > problem50.in
# ./genreel.py 100 12 75 3 8   | tee >(head -n 1 >> solutions.txt) | tail -n 75 > problem75.in
# ./genreel.py 125 12 100 3 8  | tee >(head -n 1 >> solutions.txt) | tail -n 100 > problem100.in
# ./genreel.py 150 12 125 3 8  | tee >(head -n 1 >> solutions.txt) | tail -n 125 > problem125.in
# ./genreel.py 150 12 150 3 8  | tee >(head -n 1 >> solutions.txt) | tail -n 150 > problem150.in
