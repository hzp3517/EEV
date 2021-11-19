set -e
name=$1
feature=$2
norm_features=$3
hidden_size=$4
expert_num=$5
loss_type=$6
lr=$7
run_idx=$8
gpu_ids=$9


cmd="python train_csv_moe_layers.py --dataset_mode=eev --model=baseline --gpu_ids=$gpu_ids
--log_dir=./logs/baseline_moe_layers --checkpoints_dir=./checkpoints/baseline_moe_layers --print_freq=2
--feature_set=$feature --norm_features=$norm_features
--max_seq_len=60 --gru_layers=2 --expert_num=$expert_num --hidden_size=$hidden_size
--batch_size=32 --lr=$lr --dropout_rate=0.3 --run_idx=$run_idx --verbose
--niter=30 --niter_decay=20 --num_threads=0 --loss_type=$loss_type
--name=$name --suffix={feature_set}_moe{expert_num}_hidden{hidden_size}_seq{max_seq_len}_{loss_type}_lr{lr}_run{run_idx}"
# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# --max_seq_len=60
# 官方baseline的batchsize：128，但设为128会爆显存
# 官方baseline的lr: 5e-4

# bash scripts/train_debug.sh baseline inception None 512 10 mse 5e-4 1 5
