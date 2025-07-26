#!/bin/bash


SEED=42
DIR_PATH="datasets/train/af3_1022P_1022R" 
SAVE_PATH="datasets/train/af3_1022P_1022R_aug"
CSV_PATH="datasets/train/af3_1022P_1022R.csv"
OUTPUT_CSV="datasets/train/af3_1022P_1022R_aug.csv"
NUM_AUGMENTATIONS=4

# Run the script
python src/data/preprocessing/data_augmentation.py \
    --seed $SEED \
    --dir_path $DIR_PATH \
    --save_path $SAVE_PATH \
    --csv_path $CSV_PATH \
    --output_csv $OUTPUT_CSV \
    --num_augmentations $NUM_AUGMENTATIONS