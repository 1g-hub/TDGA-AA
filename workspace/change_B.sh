#!/bin/sh

for B in 1 2 4 6 8 10 16 20 32
do
	python main.py --auto_augment=True --mag=5 --tinit=0.02 --tfin=0.02 --B=$B --prob_mul=2
done
