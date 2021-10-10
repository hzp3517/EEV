set -e
feature=$1
norm_features=$2
hidden_size=$3
loss_type=$4
lr=$5
run_idx=$6
gpu_ids=$7


cmd="python train.py --dataset_mode=eev --model=baseline --gpu_ids=$gpu_ids
--log_dir=./logs/baseline_full --checkpoints_dir=./checkpoints/baseline_full --print_freq=2
--feature_set=$feature --norm_features=$norm_features
--max_seq_len=60 --gru_layers=2 --expert_num=10 --hidden_size=$hidden_size
--batch_size=32 --lr=$lr --dropout_rate=0.3 --run_idx=$run_idx --verbose
--niter=30 --niter_decay=20 --num_threads=0 --loss_type=$loss_type
--name=baseline --suffix={feature_set}_hidden{hidden_size}_seq{max_seq_len}_{loss_type}_lr{lr}_run{run_idx}"
# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# --max_seq_len=60
# 官方baseline的batchsize：128，但设为128会爆显存
# 官方baseline的lr: 5e-4

# bash scripts/train_baseline.sh inception,vggish vggish 512,128 mse 5e-4 1 7
# bash scripts/train_baseline.sh inception None 512 mse 5e-4 1 6
# bash scripts/train_baseline.sh vggish vggish 128 mse 5e-4 1 6


# bash scripts/train_baseline.sh vggish vggish 128 mse 5e-4 1 1

# bash scripts/train_baseline.sh inception None 128 mse 1e-4 1 6