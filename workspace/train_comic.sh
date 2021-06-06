#!/usr/bin/env bash
# python main.py --dataset=comic --batch_size=32 --num_classes=6 # --exp_name=senga
# python main.py --dataset=comic --batch_size=32 --num_classes=6 --auto_augment=true --mag=5 --tinit=0.005 --tfin=0.005 --select_gamma=nonrest --allele_max=30 --fuzzy=True --mul_lambda=2 --exp_name=hoge_allele_fuzzy --pre_train_epochs=2 --Ng=2 --Np=4 --B=4 --epochs=1

# for l in 2 3 4 5 
# do
#     python main.py --dataset=comic --batch_size=32 --num_classes=6 --auto_augment=true --tinit=0.005 --tfin=0.005 --select_gamma=nonrest --allele_max=30 --fuzzy=True --mul_lambda=$l --exp_name=hoge_allele_fuzzy_lambda # --pre_train_epochs=2 --Ng=2 --Np=4 --B=4 --epochs=1
# done
python main.py --dataset=comic --batch_size=32 --num_classes=6 --auto_augment=true --tinit=0.005 --tfin=0.001 --select_gamma=nonrest --allele_max=30 --fuzzy=True --exp_name=allele_fuzzy_annealing 
# python main.py --dataset=comic --batch_size=32 --num_classes=6 --auto_augment=true --tinit=0.001 --tfin=0.0005 --select_gamma=nonrest --allele_max=30 --fuzzy=True --exp_name=allele_fuzzy_annealing
# python main.py --dataset=comic --batch_size=32 --num_classes=6 --auto_augment=true --tinit=0.0005 --tfin=0.0001 --select_gamma=nonrest --allele_max=30 --fuzzy=True --exp_name=allele_fuzzy_annealing
