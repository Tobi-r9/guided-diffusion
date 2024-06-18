#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 00:05:00
#SBATCH -C "thin"

export ENROOT_CACHE_PATH=/your/enroot/cache/path
export ENROOT_DATA_PATH=/your/enroot/data/path
source ~/.bashrc

enroot start --mount {your/project/path}:/workspace/ --rw {enroot_container_name} sh -c ' 

ITERATION="120000"
NUM_SAMPLES="12"
MODEL_FLAGS="--image_size 28 --num_channels 64 --num_res_blocks 2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
SAMPLE_FLAGS="--num_samples ${NUM_SAMPLES} --batch_size 12 --predict_xstart True --rescale_learned_sigmas False --learn_sigma False"

export PYTHONPATH=/workspace/
export OPENAI_LOGDIR={/workspace/your/logging/path}

mpirun -np 1 python scripts/image_sample.py --model_path {your/model/path} $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
python scripts/visualize.py --path ${OPENAI_LOGDIR}/samples_${NUM_SAMPLES}x28x28x1.npz'