#!/bin/sh
env="IA"
scenario="ia_simple" 
num_harvester=3
num_transporter=2
algo="mappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ./onpolicy/scripts/train/eval_ia.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_harvester ${num_harvester} --num_transporter ${num_transporter} --seed ${seed} \
    --n_rollout_threads 128 --use_wandb --episode_length 1000 --decision_dt 10.0 --num_env_steps 100000000 \
    --ppo_epoch 5 --use_wandb --hidden_size 1024 --layer_N 2 --entropy_coef 0.01 --lr 3e-4 --critic_lr 3e-4 \
    echo "training is done!"
done