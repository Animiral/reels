#!/bin/bash
# This script measures statistics about all the inputs in the input directory and puts them in a CSV table.
# All problems from the in/ directory are measured.
# For each problem:
#   - singular/full search
#   - time,examined,discovered,memorized

OUT_FILE=profall.csv

touch $OUT_FILE
# truncate -s 0 $OUT_FILE

# write headers
echo "in_file,arg,time,examined,discovered,memorized" > $OUT_FILE # trunc file for new record

# classic test suite
for f in in/*.in in/*.csv
do
	for arg in "" "--full"
	do
		echo -n "$f $arg..."

		echo -n "$f,$arg," >> $OUT_FILE
		./profile_reels.py ts $arg "$f" | tr -d '\n' >> $OUT_FILE
		echo -n ',' >> $OUT_FILE
		./reels.py -x $arg "$f" >> $OUT_FILE

		echo "DONE"
	done
done
