# 对某一个model做test
set -e

# checkpoints="baseline_9.12/baseline_inception_hidden512_seq60_mse_lr0.0005_run1"
# checkpoints="debug/baseline_inception_hidden512_seq60_mse_lr0.0005_run1"
checkpoints="baseline_9.22/baseline_inception-vggish_hidden512-128_seq60_mse_lr0.0005_run1"
name="baseline_single"

cmd="python /data8/hzp/evoked_emotion/EEV/test.py --dataset_mode=eev --model=None
--submit_dir=/data8/hzp/evoked_emotion/EEV/submit
--checkpoints_dir=/data8/hzp/evoked_emotion/EEV/checkpoints --gpu_ids=3
--name=$name --test_checkpoints='$checkpoints'"

# 这里不需要写 --write_sub_results，因为不是ensemble

# echo "\n-------------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
# echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

# bash scripts/test_single.sh