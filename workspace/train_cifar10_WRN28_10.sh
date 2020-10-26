#!/usr/bin/env bash
python main.py --auto_augment=True --tinit=0.02 --tfin=0.02 --mag=5 --prob_mul=2 --network=wresnet28_10 --epochs=300  --augment_path=./augmentation.cp
