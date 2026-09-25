#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nome.cognome@studenti.polito.it
#SBATCH --output=logs/%x_%j.out

CLASS=$1
SEED=$2
REPO=$SLURM_SUBMIT_DIR
DATA_HOME=/home/mla_group_08/TESI_EMA/datasets
RUN_NAME=skrd4ad_${CLASS}_s${SEED}
WORK=$SCRATCH_FLASH/$RUN_NAME/$SLURM_JOB_ID
DEST=$REPO/runs/$RUN_NAME

export PATH="$HOME/env_ema_tesi/bin:$PATH"
echo "Class: $CLASS | Seed: $SEED | Node: $(hostname) | Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# copia la sola classe necessaria su scratch
mkdir -p "$WORK/datasets" "$WORK/checkpoints" "$WORK/results"
cp -r "$DATA_HOME/$CLASS" "$WORK/datasets/"
cd "$WORK"

# training: legge e scrive solo su scratch
python "$REPO/main.py" \
    --epochs 200 \
    --res 3 \
    --learning_rate 0.005 \
    --batch_size 16 \
    --seed "$SEED" \
    --class_ "$CLASS" \
    --L2 2 \
    --layerloss 1 \
    --seg 1 \
    --print_epoch 10 \
    --data_path "$WORK/datasets/" \
    --ckpt_path "$WORK/checkpoints/" \
    --project_name "skrd4ad_$CLASS" \
    --net wide_res50 \
    --vis 1 \
    --aug-config "$REPO/configs/aug_legacy.json" \
    --image-size 512 \
    --image-isize 256 \
    --rate 0.05 \
    --img_path "$WORK/results/"
STATUS=$?

# riporta in home tutto tranne il dataset, poi pulisce scratch
mkdir -p "$DEST"
rsync -a --exclude datasets "$WORK/" "$DEST/" && rm -rf "$WORK"
exit $STATUS