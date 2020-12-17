#!/bin/sh

for t in 0.0001 0.001 0.002 0.003 0.005 0.01 0.02 0.03 0.04 0.05
do
	python main.py --auto_augment=True --tinit=$t --tfin=$t --mag=5 --prob_mul=2 --epochs=200
done
