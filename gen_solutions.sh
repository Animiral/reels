#!/bin/bash
# This script runs reels.py on all inputs in the input directory, looking for the minimal solution.
# Once found, appends the solution to the solution file.
# Scripts like validate.sh then use this new solution as a reference.
# The original solution is not lost and can be restored with $(head -n 1 $solution_file).

LIMITS="--timeout=60 --memsize=1000000"

# classic test suite
for f in in/*.in
do
	echo -n "$(basename $f) ..."
	ref_file=$(echo $f | sed 's#/\(.*\)\.in#/s_\1.txt#')
	./reels.py $LIMITS --out_file "$ref_file" "$f"
	echo "DONE"
done

# CSV test suite
for f in in/*.csv
do
	echo -n "$(basename $f) ..."
	ref_file=$(echo $f | sed 's#/\(.*\)\.csv#/s_\1.txt#')
	./reels.py $LIMITS --out_file "$ref_file" "$f"
	echo "DONE"
done
