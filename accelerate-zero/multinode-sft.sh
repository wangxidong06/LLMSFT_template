#!/bin/bash
#SBATCH -N 12 --gres=gpu:4 --qos=gpugpu
module purge
module load anaconda compilers/cuda/11.6 compilers/gcc/9.3.0 cudnn/8.4.0.27_cuda11.x
source activate deepspeed 
export NCCL_ALGO=Ring
# export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export PYTHONUNBUFFERED=1

#python *.py

experiment_name=llama_70B_sft_testfinal1
model_dir=/your/model/dir/in/huggingface/mode
data_file=./data/response_data_list.json
output_dir=./ckpts
log_folder="./logs/${SLURM_JOB_ID}"
mkdir -p $log_folder


run_name=llama-70-v5.0
host=($(scontrol show hostnames))
for hostname in "${host[@]}"; do
	echo $hostname >> ${log_folder}/${SLURM_JOB_ID}.log
done

process_port=29501

# 在 12 个节点上循环运行 srun
for rank in {0..11}
do
	srun -N 1 --gres=gpu:4 -w ${host[${rank}]} \
	accelerate launch \
		--config_file ./configs/finetune.yaml \
		--num_processes 48 \
		--num_machines 12 \
		--machine_rank ${rank} \
		--main_process_ip "${host[0]}" \
		--main_process_port ${process_port} \
		--num_cpu_threads_per_process 4 \
		--deepspeed_multinode_launcher standard ./src/sft.py \
		--model_path ${model_dir} \
		--experiment_name ${experiment_name} \
		--gradient_accumulation_steps 6 \
		--data_dir  ${data_file} \
		--output_dir ${output_dir} \
		--log_dir ./logs_wandb \
		--n_epochs 1 \
		--train_bsz_per_gpu 1 \
		--eval_bsz_per_gpu 1 \
		--learning_rate 5e-5 \
		--eval_step -1 \
		--save_step -1 \
		--max_seq_len 2048 \
		--warmup_rates 0.03 \
		--max_ckpts 1 \
		--gradient_checkpointing  >> ${log_folder}/rank${rank}.log 2>&1 &
done

# 等待所有后台任务完成
wait