#!/bin/sh
#SBATCH -A IscrC_GenOpt
#SBATCH -p boost_usr_prod
#SBATCH --time=14:00:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=8      
#SBATCH --job-name=finetune

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_GenOpt/giordano/VAST
export WANDB_MODE=offline
module load anaconda3
source activate triangle

config_name='pretrain_triangle/downstream/finetunearea256batchlossonlyarea4Mod120k'
output_dir=./output/triangle/$config_name

save_dir=./output/triangle/pretrain_vast/
### VIDEO-RET


#retrieval-msrvtt
srun python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/triangle/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir $output_dir \
--output_dir $save_dir/downstream/finetuneMSRVTT4 \
