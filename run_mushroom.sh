#!/bin/bash 

for seed in {0..3}; do 
    python ./src/mlp_mc_coldstart.py --dataset mushroom --method joint_bald --lr 1e-2 --bs 64 --seed $seed --nepoch 500 --qs 5 --steps 40 --beta 0.75 & 
    python ./src/mlp_mc_coldstart.py --dataset mushroom --method e_bald --lr 1e-2 --bs 64 --seed $seed --nepoch 500 --qs 5 --steps 40 --beta 0.75 & 
    python ./src/mlp_mc_coldstart.py --dataset mushroom --method random --lr 1e-2 --bs 64 --seed $seed --nepoch 500 --qs 5 --steps 50 --beta 0.75 & 
    python ./src/mlp_mc_coldstart.py --dataset mushroom --method joint_bald_ucb --lr 1e-2 --bs 64 --seed $seed --nepoch 500 --qs 5 --steps 40 --beta 0.75 & 
    python ./src/mlp_mc_coldstart.py --dataset mushroom --method joint_bald_ts --lr 1e-2 --bs 64 --seed $seed --nepoch 500 --qs 5 --steps 40 --beta 0.75 & 
    python ./src/mlp_mc_coldstart.py --dataset mushroom --method naive_bald --lr 1e-2 --bs 64 --seed $seed --nepoch 500 --qs 5 --steps 80 --beta 0.75 & 
    wait 
done
