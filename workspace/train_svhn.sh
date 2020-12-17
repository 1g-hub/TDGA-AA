#!/usr/bin/env bash
python main.py --auto_augment=True --network=wresnet28_10 --mag=5 --tinit=0.02 --tfin=0.02 --prob_mul=3 --lr=0.005 --weight_decay=0.005 --dataset=svhn-core --train_split=trainval --epochs=400
