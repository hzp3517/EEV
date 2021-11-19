set -e
name=$1
feature=$2
norm_features=$3
tcn_channels=$4
hidden_size=$5
loss_type=$6
lr=$7
run_idx=$8
gpu_ids=$9


cmd="python train.py --dataset_mode=eev --model=eev1st --gpu_ids=$gpu_ids
--log_dir=./logs/debug --checkpoints_dir=./checkpoints/debug --print_freq=2
--feature_set=$feature --norm_features=$norm_features
--tcn_channels=$tcn_channels --hidden_size=$hidden_size
--batch_size=32 --lr=$lr --dropout_rate=0.2 --run_idx=$run_idx --verbose
--niter=1 --niter_decay=1 --num_threads=0 --loss_type=$loss_type
--name=$name --suffix={feature_set}_tcn{tcn_channels}_hidden{hidden_size}_{loss_type}_lr{lr}_run{run_idx}"
# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# --log_dir=./logs/eev1st_subset_5 --checkpoints_dir=./checkpoints/eev1st_subset_5
# --max_seq_len=60

# bash scripts/train_eev1st.sh eev1st trill_distilled trill_distilled 512 128 mse 5e-3 1 5