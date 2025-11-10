#!/bin/bash
#SBATCH --account=cis240124p
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH --job-name=kmeans_cli
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Create logs dir if doesn't already exist
mkdir -p logs

echo "Job started at $(date)"
echo "This is being ran from $(pwd)"

export R_LIBS_USER=~/R/workshop_libs

Rscript workshop_handson_Kmeans.R

echo "Job completed at $(date)"
