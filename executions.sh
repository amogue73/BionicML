#!/bin/bash

algorithms=("GA" "PSO" "ACO" "WOA" "GWO")
for m in {1..20}
do
    for i in {10..30..5}
    do
        for j in {10..30..5}
        do
            for k in 1 2 4 6 8
            do
                for l in "${algorithms[@]}"
                do
                    python3 algorithms3e24.py -m $l $i $j $k 100
                done
            done
        done
    done
done