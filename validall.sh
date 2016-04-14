#!/bin/bash
# This script tests all reels.py outputs against their solutions using validate.py.
# All problems from the in/ directory are tested.

for f in in/*.in
do
	out=$(echo $f | sed 's#\.in#\.out#')
	$(./reels.py --out_file "$out" "$f")
	r=$(echo $f | sed 's#/\(.*\)\.in#/s_\1.txt#')
	ref=$(cat $r)

	echo $(basename $f) : $(./validate.py --out_file="$out" --solution="$ref" "$f")
done
