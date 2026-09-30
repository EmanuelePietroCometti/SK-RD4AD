#!/bin/bash
# un job per dataset (in parallelo tra loro), seed eseguiti in sequenza nel job
mkdir -p logs
SEEDS="0 1 2 42 101"
for c in carpet tessuto_nero tessuto_nero_dust_validation tessuto_nero_dust_train; do
    sbatch --job-name="skrd_${c}" run_experiment.sbatch "$c" $SEEDS
done
