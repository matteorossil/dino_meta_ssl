#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=1M_deblur_full
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

module purge

singularity exec --nv \
	--overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u eval_linear.py \
												--pretrained_weights /vast/mr6744/pretrained_models_dino/1M_deblur/checkpoint0010.pth \
												--epochs 100 \
												--output_dir /vast/mr6744/linear_evaluation_dino/1M_deblur \
												--data_path /vast/mr6744/MiniImagenet \
												--num_labels 1000 "