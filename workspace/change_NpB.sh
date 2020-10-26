#!/bin/sh

for B in 2 4 8 16 32
do
    for Np in 16 32 64 128
    do
        python main.py --auto_augment=True --tinit=0.03 --tfin=0.03 --mag=5 --B=$B --Np=$Np
	done
done
