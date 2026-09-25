#!/bin/bash
mkdir -p logs
for c in dataset_tessuto_nero tessuto_nero_dust_train tessuto_nero_dust_validation; do
    for s in 0 1 2 42 101; do
        sbatch --job-name="skrd_${c}_s${s}" run_experiment.sbatch "$c" "$s"
    done
done