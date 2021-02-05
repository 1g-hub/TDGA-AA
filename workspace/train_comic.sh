#!/usr/bin/env bash
python main.py --dataset=comic --batch_size=32 --num_classes=6 # --exp_name=senga
# python main.py --dataset=comic --batch_size=32 --num_classes=6 --auto_augment=true --mag=5 --tinit=0.005 --tfin=0.005
