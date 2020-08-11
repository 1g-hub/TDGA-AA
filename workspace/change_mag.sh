#!/bin/sh

for mag in 0 5 10 15 20 25 30
do
	python main.py --epochs=200 --optimizer=sgd --auto_augment=True --mag=$mag
done
