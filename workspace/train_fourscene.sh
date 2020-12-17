#!/usr/bin/env bash
python main.py --dataset=fourscene-comic --ckpt_path=model.pt --num_classes=6 --auto_augment=true --mag=5 --tinit=0.001 --tfin=0.001