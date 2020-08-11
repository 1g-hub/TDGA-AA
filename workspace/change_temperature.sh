#!/bin/sh

for t in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08
do
	python main.py --epochs=200 --optimizer=sgd --auto_augment=True --tinit=$t --tfin=$t --mag=4
done
