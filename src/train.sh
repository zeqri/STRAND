#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --mem=400gb 




source /home/fr/fr_fr/fr_ma453/ENTER/bin/activate diffdock

workdir=/home/fr/fr_fr/fr_ma453/folder/DiffDock-PR

cd "$workdir"
NUM_FOLDS=1  # number of seeds to try, default 5
SEED=0  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=2
BATCH_SIZE=12  # split across all GPUs

NAME="rna"  # change to name of config file
RUN_NAME="classic_run_aug_3db" # should uniauely describe the current experiment
CONFIG="config/${NAME}.yaml"

SAVE_PATH="ckpts/${RUN_NAME}"
VISUALIZATION_PATH="visualization/${RUN_NAME}"

echo SAVE_PATH: $SAVE_PATH

python src/main.py \
    --mode "train" \
    --config_file $CONFIG \
    --run_name $RUN_NAME \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --num_folds $NUM_FOLDS \
    --num_gpu $NUM_GPU \
    --gpu $CUDA --seed $SEED \
    --logger "wandb" \
    --project "DiffDock Tuning" \
    --visualize_n_val_graphs 0 \
    --visualization_path $VISUALIZATION_PATH \
    --tr_weight 0.5 \
    --rot_weight 0.5\
    --knn_size 30\
    --patience 250\
    --nv 4 --ns 16 --max_radius 5 \
    --recache \
    --data_path "/gpfs/bwfor/work/ws/fr_ma453-data_alpha_fold/3db_reflection_tr"\
    --data_file "/gpfs/bwfor/work/ws/fr_ma453-data_alpha_fold/csv/3db_1022P_1022R_tr_ref_aug.csv"\
    

    # --data_file "/home/fr/fr_fr/fr_ma453/folder/DiffDock-PR/datasets/af3/af3_1022P_1022R_rot_mod.csv"\
    # --data_path "/gpfs/bwfor/work/ws/fr_ma453-data_alpha_fold/dill_mod_rot"\


    # --recache \
    #--checkpoint_path $SAVE_PATH \
    #--debug True # load small dataset
    #--entity coarse-graining-mit \

# if you accidentally screw up and the model crashes
# you can restore training (including optimizer)
# by uncommenting --checkpoint_path