#!/bin/bash

if [ ! -d "results" ]; then
    mkdir results
fi

for i in {0..9}
do
    if [ ! -d "results/ex0$i" ]; then
        mkdir "results/ex0$i"
    fi
done

for i in {0..9}
do
    python3 ex0$i/test.py > results/ex0$i/result.txt
done
