#!/bin/sh

# for t in 0.0001 0.001 0.002 0.003 0.005 0.01 0.02 0.03 0.04 0.05
for t in 0.005 0.006 0.007 0.008 0.009 0.01
do
    python main.py --dataset=comic --batch_size=32 --num_classes=6 --auto_augment=true --mag=5 --tinit=$t --tfin=$t --exp_name=rest5 # --pre_train_epochs=2 --Ng=2 --Np=4 --B=4 --epochs=1
done
