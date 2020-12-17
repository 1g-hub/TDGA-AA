#!/bin/sh

for mag in 0 1 2 3 4 5 6 7 8 9 10
do
	python main.py --auto_augment=True --tinit=0.02 --tfin=0.02 --mag=$mag --prob_mul=2 --network=wresnet28_10 --epochs=200
done
