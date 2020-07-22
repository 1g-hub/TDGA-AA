#!/bin/sh

for mag in 0 5 10 15 20 25 30
do
	python main.py --seed=24 --optimizer=sgd --auto_augment=True --mag=$mag
done
