#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 00:03:00
#SBATCH -C "thin"

export ENROOT_CACHE_PATH=/proj/berzelius-2021-89/users/x_tohop/enroot/cache
export ENROOT_DATA_PATH=/proj/berzelius-2021-89/users/x_tohop/enroot/data
source ~/.bashrc

enroot start --mount /proj/berzelius-2021-89/users/x_tohop/guided-diffusion:/workspace/ --rw nvidia_pytorch_22.09 sh -c 'cd /workspace/guided-diffusion; 


MODEL_FLAGS="--image_size 28 --num_channels 64 --num_res_blocks 2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
SAMPLE_FLAGS="--num_samples 10 --batch_size 10 --predict_xstart=True --rescale_learned_sigmas=False --learn_sigma False"

export PYTHONPATH=/workspace/
export OPENAI_LOGDIR=/workspace/samples/test

mpirun -np 1 python scripts/image_sample.py --model_path models/test/ema_0.9999_020000.pt $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS'