#!/bin/bash

NUM_FOLDS=1  # number of seeds to try, default 5
SEED=0  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=1
BATCH_SIZE=1  # split across all GPUs
NUM_SAMPLES=40
NAME="rna_inf"  # change to name of config file
CONFIG="config/${NAME}.yaml"

FILTERING_PATH="models/CONF_MODEL/fold_0" #trained models

# =============================
# Dataset selection
# =============================
DATASET=$1  # first argument (rnapro, xray, nonxray)

if [ "$DATASET" == "rnapro" ]; then
    Data_path="datasets/test_data/rnapro/model_predictions"
    Data_file="datasets/test_data/rnapro/rnapro.csv"
    GROUND_TRUTH_PATH="datasets/test_data/rnapro/ground_truth"
elif [ "$DATASET" == "xray" ]; then
    Data_path="datasets/test_data/xray_af3/model_predictions"
    Data_file="datasets/test_data/xray_af3/af3_xray.csv"
    GROUND_TRUTH_PATH="datasets/test_data/xray_af3/ground_truth"
elif [ "$DATASET" == "nonxray" ]; then
    Data_path="datasets/test_data/non_xray_af3/model_predictions"
    Data_file="datasets/test_data/non_xray_af3/af3_non_xray.csv"
    GROUND_TRUTH_PATH="datasets/test_data/non_xray_af3/ground_truth"
else
    echo "Invalid dataset argument! Please use: rnapro, xray, or nonxray"
    exit 1
fi

# Define arrays for the different configurations
declare -a SCORE_PATHS=(
    "models/STRAND_TR/fold_0"
    "models/STRAND_ROT/fold_0"
    "models/STRAND_TR_ROT/fold_0"
)

declare -a RUN_NAMES=(
    "STRAND_TR"
    "STRAND_ROT"
    "STRAND_TR_ROT"
)

declare -a VISUALIZATION_PATHS=(
    "visualization/STRAND_TR"
    "visualization/STRAND_ROT"
    "visualization/STRAND_TR_ROT"
)

# Loop through each configuration
for i in "${!SCORE_PATHS[@]}"; do
    SCORE_PATH="${SCORE_PATHS[$i]}"
    RUN_NAME="${RUN_NAMES[$i]}"
    VISUALIZATION_PATH="${VISUALIZATION_PATHS[$i]}"
    STORAGE_PATH="storage/${RUN_NAME}.pkl"
    
    echo "==================================="
    echo "Running configuration $((i+1))/3"
    echo "RUN_NAME: $RUN_NAME"
    echo "SCORE_MODEL_PATH: $SCORE_PATH"
    echo "CONFIDENCE_MODEL_PATH: $FILTERING_PATH"
    echo "VISUALIZATION_PATH: $VISUALIZATION_PATH"
    echo "GROUND_TRUTH_PATH: $GROUND_TRUTH_PATH"
    echo "STORAGE_PATH: $STORAGE_PATH"
    echo "==================================="
    
    python src/main_inf.py \
        --mode "test" \
        --config_file $CONFIG \
        --run_name $RUN_NAME \
        --batch_size $BATCH_SIZE \
        --num_folds $NUM_FOLDS \
        --num_gpu $NUM_GPU \
        --gpu $CUDA --seed $SEED \
        --logger "wandb" \
        --project "STRAND Tuning" \
        --visualize_n_val_graphs 100 \
        --visualization_path $VISUALIZATION_PATH \
        --filtering_model_path $FILTERING_PATH \
        --score_model_path $SCORE_PATH \
        --num_samples $NUM_SAMPLES \
        --prediction_storage $STORAGE_PATH \
        --knn_size 30 \
        --data_path $Data_path \
        --data_file $Data_file\
        --run_inference_without_confidence_model #flag to use confidence model or not    

    python src/visualize_inf_manual.py\
        --gt_path $GROUND_TRUTH_PATH \
        --samples_path $VISUALIZATION_PATH\
        --exp_name  $RUN_NAME 


    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Configuration $RUN_NAME completed successfully"
    else
        echo "Configuration $RUN_NAME failed with exit code $?"
        # Uncomment the next line if you want to stop on first failure
        # exit 1
    fi
    
    echo ""
done

echo "All configurations completed!"