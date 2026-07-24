#!/bin/bash -l
#SBATCH --job-name=led_synchronizer
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gres=gpu:1                            # requests 1 GPU (unused, but required on this GPU-only cluster)
#SBATCH --time=24:00:00

set -euo pipefail

# ---- Log file named by start time ----
# SLURM's --output/--error only support fixed tokens (%j job id, %x job name, etc), not
# timestamps, so the log path is built here at runtime and stdout/stderr redirected manually
# instead of via #SBATCH.
LOG_DIR="/scratch/picek/savedir/logs/sync"
mkdir -p "$LOG_DIR"
LOG_BASENAME="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
exec > "${LOG_DIR}/${LOG_BASENAME}.out" 2> "${LOG_DIR}/${LOG_BASENAME}.err"

# ---- User-configurable settings ----
CONDA_ENV_NAME="sync_pipeline"                        # TODO
REPO_DIR="/home/picek/repos/MultimodalBrainFeaturePipeline/data_synchronization"     # TODO: .../MultimodalBrainFeaturePipeline/data_synchronization

VIDEO_FOLDER="/home/picek/nas_mount/projects/upperlimb/__UP2/00_RAW_DATA/UP2004"                     # TODO: same videos as the get_LED_signal job
LED_SIGNALS_FOLDER="/work/upcourtine/multimodal/data/UP2004/led_positions"   # TODO: = OUTPUT_FOLDER from the get_LED_signal job
ECOG_FOLDER="/home/picek/nas_mount/projects/upperlimb/__UP2/2_MERGED_TMP_ARCHIVE/01_SMR2NWB_PROCESSED/UP2004"                   # TODO
LOG_TABLE_PATH="/work/upcourtine/multimodal/data/UP2004/sync.csv"             # TODO
SYNC_IMAGES_PATH="/work/upcourtine/multimodal/data/UP2004/sync_output"    # TODO
EXTENSION=".mkv"                                          # change to ".mkv" if needed

# ---- Mount the server ----
mount_nas.sh -r # the script has to be in the PATH, e.g. in /home/<USER>/bin

# ---- Activate conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

cd "$REPO_DIR"

echo "Job started: $(date)"
echo "Node: $(hostname)"

python LEDSynchronizer.py \
    "$VIDEO_FOLDER" \
    "$LED_SIGNALS_FOLDER" \
    "$ECOG_FOLDER" \
    "$LOG_TABLE_PATH" \
    "$SYNC_IMAGES_PATH" \
    "$EXTENSION"

unmount_nas.sh # the script has to be in the PATH, e.g. in /home/<USER>/bin

echo "Job finished: $(date)"
