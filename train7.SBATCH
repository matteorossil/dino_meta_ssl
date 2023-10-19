#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=D_200K_blur_full
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

module purge

srun \
	singularity exec --nv \
	--overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u main_dino.py \
													--arch vit_small \
													--epochs 300 \
                                                    --saveckp_freq 5 \
													--optimizer adamw \
                                                    --teacher_temp 0.07 \
                                                    --warmup_teacher_temp_epochs 30 \
                                                    --norm_last_layer false \
													--output_dir /vast/mr6744/pretrained_models_dino/200K_deblur_new_full \
													--data_path /vast/mr6744/SAYCAM_large_deblur "