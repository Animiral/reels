#!/bin/bash
# Timings of reels.py

DUMP=timings.out

date >> $DUMP

for f in in/*.in
do
	echo $(basename $f) : $(./profile_reels.py ts "$f") | tee -a $DUMP
done

for f in in/*.csv
do
	echo $(basename $f) : $(./profile_reels.py ts "$f") | tee -a $DUMP
done
