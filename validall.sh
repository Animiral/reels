#!/bin/bash
# This script tests all reels.py outputs against their solutions using validate.py.
# All problems from the in/ directory are tested.

LIMITS="--timeout=60 --memsize=1000000"

# classic test suite
for f in in/*.in
do
	out=$(echo $f | sed 's#\.in#\.out#')
	$(./reels.py $LIMITS --out_file "$out" "$f")
	r=$(echo $f | sed 's#/\(.*\)\.in#/s_\1.txt#')
	ref=$(cat $r)

	if [ ! -f "$out" ]; then sleep 1; fi
	echo -n "$(basename $f) : "
	./validate.py --out_file="$out" --solution="$ref" "$f"
done

# CSV test suite
for f in in/*.csv
do
	out=$(echo $f | sed 's#\.csv#\.out#')
	$(./reels.py $LIMITS --out_file "$out" "$f")
	r=$(echo $f | sed 's#/\(.*\)\.csv#/s_\1.txt#')
	ref=$(cat $r)

	if [ ! -f "$out" ]; then sleep 1; fi
	echo -n "$(basename $f) : "
	./validate.py --out_file="$out" --solution="$ref" "$f"
done
