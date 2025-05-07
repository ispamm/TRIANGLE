#!/bin/sh
#SBATCH -A IscrC_NeuroGen
#SBATCH -p boost_usr_prod
#SBATCH --time=23:00:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=8      
#SBATCH --job-name=tst_youcook

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_NeuroGen/luigi/TRIANGLE
export WANDB_MODE=offline
module load anaconda3
source activate triangle

# config_name='pretrain_triangle'
# output_dir=./output/triangle/$config_name

### VIDEO-RET

#retrieval-youcook
srun python3 -m torch.distributed.launch  ./run.py \
--config ./config/triangle/finetune_cfg/retrieval-youcook.json \
--pretrain_dir /leonardo_scratch/fast/IscrC_NeuroGen/luigi/TRIANGLE/downstream/finetune_youcook/ \
--mode testing