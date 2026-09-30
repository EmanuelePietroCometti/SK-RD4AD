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
#SBATCH --mail-user=s346291@studenti.polito.it
#SBATCH --output=logs/%x_%j.out

# uso: sbatch run_experiment.sbatch <CLASS> [SEED ...]
# un job per dataset: le run dei vari seed vengono eseguite in sequenza
CLASS=$1
shift
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 42 101)

REPO=$SLURM_SUBMIT_DIR
DATA_HOME=/home/mla_group_08/TESI_EMA/datasets
WORK=$SCRATCH_FLASH/skrd4ad_${CLASS}/$SLURM_JOB_ID

export PATH="$HOME/env_ema_tesi/bin:$PATH"
echo "Class: $CLASS | Seeds: ${SEEDS[*]} | Node: $(hostname) | Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# copia la sola classe necessaria su scratch, una volta per tutte le run
mkdir -p "$WORK/datasets"
cp -r "$DATA_HOME/$CLASS" "$WORK/datasets/"

FAILED=()
for SEED in "${SEEDS[@]}"; do
    RUN_NAME=skrd4ad_${CLASS}_s${SEED}
    RUN_WORK=$WORK/runs/$RUN_NAME
    DEST=$REPO/runs/$RUN_NAME

    echo "=========================================================="
    echo "Starting run -> Class: $CLASS | Seed: $SEED | $(date)"
    echo "=========================================================="

    mkdir -p "$RUN_WORK/checkpoints" "$RUN_WORK/results"
    cd "$RUN_WORK"

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
        --ckpt_path "$RUN_WORK/checkpoints/" \
        --project_name "skrd4ad_$CLASS" \
        --net wide_res50 \
        --vis 1 \
        --aug-config "$REPO/configs/aug_legacy.json" \
        --image-size 512 \
        --image-isize 256 \
        --rate 0.05 \
        --img_path "$RUN_WORK/results/"
    STATUS=$?
    [ $STATUS -ne 0 ] && FAILED+=("$SEED")

    # riporta in home la run appena conclusa, così non si perde se il job va in timeout
    mkdir -p "$DEST"
    rsync -a "$RUN_WORK/" "$DEST/" && rm -rf "$RUN_WORK"

    echo "Finished run -> Class: $CLASS | Seed: $SEED | status: $STATUS | $(date)"
done

# pulisce scratch (dataset incluso)
cd "$REPO"
rm -rf "$WORK"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Run fallite per i seed: ${FAILED[*]}"
    exit 1
fi
echo "Tutte le run per $CLASS completate"
