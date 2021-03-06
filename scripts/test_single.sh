# 对某一个model做test
set -e

# checkpoints="baseline_9.12/baseline_inception_hidden512_seq60_mse_lr0.0005_run1"
# checkpoints="debug/baseline_inception_hidden512_seq60_mse_lr0.0005_run1"
checkpoints="baseline_full/baseline_inception_hidden128_seq60_mse_lr0.0001_run1"
name="baseline_single"

cmd="python /data8/hzp/evoked_emotion/EEV/test.py --dataset_mode=eev --model=baseline
--submit_dir=/data8/hzp/evoked_emotion/EEV/submit
--checkpoints_dir=/data8/hzp/evoked_emotion/EEV/checkpoints --gpu_ids=1
--name=$name --test_checkpoints='$checkpoints'"

# 这里不需要写 --write_sub_results，因为不是ensemble
# model那里如果不写对应的模型，checkpoints目录中就不会记录test_opt.txt文件

# echo "\n-------------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
# echo "-------------------------------------------------------------------------------------\n"


# bash scripts/test_single.sh

echo $cmd | sh