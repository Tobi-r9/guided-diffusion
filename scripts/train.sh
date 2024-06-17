#!/bin/bash
#SBATCH --gpus 8
#SBATCH -t 12:00:00
#SBATCH -C "thin"

export ENROOT_CACHE_PATH=/proj/berzelius-2021-89/users/x_tohop/enroot/cache
export ENROOT_DATA_PATH=/proj/berzelius-2021-89/users/x_tohop/enroot/data
source ~/.bashrc

enroot start --mount /proj/berzelius-2021-89/users/x_tohop/guided-diffusion:/workspace/ --rw nvidia_pytorch_22.09 sh -c 'cd /workspace/guided-diffusion; 


MODEL_FLAGS="--image_size 28 --num_channels 64 --num_res_blocks 2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 256 --predict_xstart=True --rescale_learned_sigmas=False --learn_sigma False"

export PYTHONPATH=/workspace/
export OPENAI_LOGDIR=/workspace/models/test

mpirun -np 8 python scripts/image_train.py --data_dir datasets/imbalanced_7-0.99_0-0.99.zip $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS'