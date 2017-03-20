#!/bin/bash

g++ pitch_compare.cpp -o pitch_compare

python pitch_autocorr.py FILELIST

./pitch_compare FILELIST
