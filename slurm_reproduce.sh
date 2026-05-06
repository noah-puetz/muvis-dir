#!/bin/bash
#SBATCH --job-name=muvis-dir
#SBATCH --array=0-539
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu           # adjust to your cluster
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# ── required ────────────────────────────────────────────────────────────────
DATA_DIR=${DATA_DIR:?'Set DATA_DIR before submitting, e.g. sbatch --export=DATA_DIR=/path/to/data slurm_reproduce.sh'}
OUTPUT_ROOT=${OUTPUT_ROOT:-predictions}
# ────────────────────────────────────────────────────────────────────────────

SEEDS=(42 43 44 45 46 47 48 49 50 51)
DATASETS=(
    BeijingPM10Quality
    BeijingPM25Quality
    PPGDalia
    Panasonic18650PFData
    "REVS/2013_Monterey_Motorsports_Reunion"
    "REVS/2013_Targa_Sixty_Six"
    "REVS/2014_Targa_Sixty_Six"
    TennesseeEastmanProcess
    VehicleDynamicsDataset
)
METHODS=(lds sqinv focal_l1 conr rnc uvote)

N_DATASETS=${#DATASETS[@]}
N_METHODS=${#METHODS[@]}

# map flat index → (seed, dataset, method)
i=$SLURM_ARRAY_TASK_ID
seed_idx=$(( i / (N_DATASETS * N_METHODS) ))
ds_idx=$(( (i / N_METHODS) % N_DATASETS ))
m_idx=$(( i % N_METHODS ))

SEED=${SEEDS[$seed_idx]}
DATASET=${DATASETS[$ds_idx]}
METHOD=${METHODS[$m_idx]}

mkdir -p logs "$OUTPUT_ROOT/seed_${SEED}"

echo "[$SLURM_ARRAY_TASK_ID] seed=$SEED  dataset=$DATASET  method=$METHOD"

python -m muvis_dir.train \
    --dataset  "$DATASET" \
    --method   "$METHOD" \
    --seed     "$SEED" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_ROOT/seed_${SEED}"
