#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --job-name=medlord_ldm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=32gb


# Log compute-node infos into slurm log
echo '***********************************'
echo "Job placed on:" $HOSTNAME
echo '***********************************'
echo "GPU infos"
nvidia-smi
echo '***********************************'
echo 'CPU infos'
lscpu | grep 'Model name'
lscpu | grep 'Architecture'
cat /proc/meminfo | grep 'MemTotal'
cat /proc/meminfo | grep 'MemFree'
echo '***********************************'

# Load modules
echo '***********************************'
module purge
module load devel/miniforge
module list
echo '***********************************'

# Slurm environment info
echo '***********************************'
echo 'Slurm environment:'
printenv | grep SLURM_JOB_NAME
printenv | grep SLURM_JOB_GPUS
printenv | grep SLURM_JOB_ID
printenv | grep SLURM_JOB_USER
printenv | grep SLURM_JOB_PARTITION
printenv | grep SLURM_MEM_PER_NODE
echo '***********************************'

# -----------------------------------------------
# Paths — adjust to match your storage system
# -----------------------------------------------
REPO_DIR=/path/to/medlord          # root of this repository
TRAINING_IDS=/path/to/train/latents.csv
VALIDATION_IDS=/path/to/val/latents.csv
CONFIG=${REPO_DIR}/configs/diffusion/medlord_new.yaml
OUTPUT_DIR=/path/to/outputs
RUN_NAME=ldm_v1
CONDA_ENV=/path/to/conda/envs/medlord

# -----------------------------------------------
echo 'repo dir:' ${REPO_DIR}
echo 'training ids:' ${TRAINING_IDS}
echo 'validation ids:' ${VALIDATION_IDS}
echo 'config:' ${CONFIG}
echo 'output dir:' ${OUTPUT_DIR}

# Show Python version
echo '***********************************'
echo 'Show Python version and available libraries'
${CONDA_ENV}/bin/python --version
${CONDA_ENV}/bin/pip list | grep torch
echo '***********************************'

mkdir -p logs

cd ${REPO_DIR}

# -----------------------------------------------
# Run training (single GPU)
# -----------------------------------------------
${CONDA_ENV}/bin/python src/scripts/train_ldm.py \
    --config ${CONFIG} \
    --training_ids ${TRAINING_IDS} \
    --validation_ids ${VALIDATION_IDS} \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${RUN_NAME} \
    --seed 42

echo 'python script executed'
