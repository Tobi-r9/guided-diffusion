#!/bin/bash
#SBATCH --gpus 8
#SBATCH -t 12:00:00
#SBATCH -C "thin"

export ENROOT_CACHE_PATH=/your/enroot/cache/path
export ENROOT_DATA_PATH=/your/enroot/data/path
source ~/.bashrc

enroot start --mount {your/project/path}:/workspace/ --rw {enroot_container_name} sh -c ' 


MODEL_FLAGS="--image_size 28 --num_channels 64 --num_res_blocks 2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 256 --predict_xstart=True --rescale_learned_sigmas=False --learn_sigma False"

export PYTHONPATH=/workspace/
export OPENAI_LOGDIR={/workspace/your/logging/path}

mpirun -np 8 python scripts/image_train.py --data_dir {your/data/path} $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS'