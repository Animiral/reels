#!/bin/bash
# This script compiles estimate.cpp into a library to be accessed by reels.py.

CXX="g++ --std=c++14"
CXXFLAGS="-g -Wall"
# CXXFLAGS="-g -Wall -DLOG_ENABLED=true"

function make_so {
	$CXX $CXXFLAGS -shared -fPIC -o estimate.so estimate.cpp   # so/library
}

function make_bin {
	$CXX $CXXFLAGS -o estimate estimate.cpp     # self-contained unit-test binary
}

make_so && make_bin
