set -e
feature=$1
norm_features=$2
hidden_size=$3
lr=$4
run_idx=$5
gpu_ids=$6


cmd="python train.py --dataset_mode=eev --model=baseline --gpu_ids=$gpu_ids
--log_dir=./logs/test --checkpoints_dir=./checkpoints/test --print_freq=2
--feature_set=$feature --norm_features=$norm_features
--max_seq_len=60 --gru_layers=2 --expert_num=10 --hidden_size=$hidden_size
--batch_size=128 --lr=$lr --dropout_rate=0.3 --run_idx=$run_idx --verbose
--niter=30 --niter_decay=20 --num_threads=0 --loss_type=mse
--name=baseline --suffix={feature_set}_hidden{hidden_size}_seq{max_seq_len}_lr{lr}_run{run_idx}"
# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# --max_seq_len=60
# 官方baseline的batchsize：128
# 官方baseline的lr: 5e-4
# bash scripts/train_baseline.sh inception,vggish vggish 512,128 5e-4 1 4
