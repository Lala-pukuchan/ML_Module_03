#!/bin/bash

# make results directory
if [ ! -d "results" ]; then
    mkdir results
fi

# make results/ex directory
for i in {0..9}
do
    if [ ! -d "results/ex0$i" ]; then
        mkdir "results/ex0$i"
    fi
done

# get current directory
currentDir=$(pwd)

# Export the new path to PYTHONPATH
export PYTHONPATH="$currentDir:$PYTHONPATH"

# execute test.py
for i in {0..9}
do
    if [[ $i -ne 7 ]]; then
        python3 ex0$i/test.py > results/ex0$i/result.txt
    else
        # python3 ex07/mono_log.py --zipcode=4 > results/ex07/result_mono.txt
        python3 ex07/mono_log.py --zipcode=1 > results/ex07/result_mono.txt
        python3 ex07/multi_log.py > results/ex07/result_multi.txt
    fi
done
