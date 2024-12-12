#!/bin/bash

python ./src/mlp_mc_coldstart_syn.py --dataset synthetic_circle --method naive_bald --lr 1e-2 --bs 8 --seed 3 --nepoch 600 --qs 5 --steps 100 --beta 0.75;
python ./src/mlp_mc_coldstart_syn.py --dataset synthetic_circle --method e_bald --lr 1e-2 --bs 8 --seed 3 --nepoch 600 --qs 5 --steps 30 --beta 0.75;
python ./src/mlp_mc_coldstart_syn.py --dataset synthetic_circle --method joint_bald --lr 1e-2 --bs 8 --seed 3 --nepoch 800 --qs 5 --steps 40 --beta 0.75;
python ./src/mlp_mc_coldstart_syn.py --dataset synthetic_circle --method joint_bald_ucb --lr 1e-2 --bs 8 --seed 3 --nepoch 500 --qs 5 --steps 40 --beta 0.75;
python ./src/mlp_mc_coldstart_syn.py --dataset synthetic_circle --method joint_bald_ts --lr 1e-2 --bs 8 --seed 3 --nepoch 500 --qs 5 --steps 40 --beta 0.75;
