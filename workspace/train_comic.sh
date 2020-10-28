#!/usr/bin/env bash
python main.py --dataset=comic --batch_size=32 --num_classes=6 --train_split=trainval --auto_augment=true --B=8
