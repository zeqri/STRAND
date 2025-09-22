NUM_FOLDS=1  # number of seeds to try, default 5
SEED=0  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=1
BATCH_SIZE=12  # split across all GPUs

NAME="rna"  # change to name of config file
RUN_NAME="STRAND_TR_ROT" # should uniauely describe the current experiment
CONFIG="config/${NAME}.yaml"

SAVE_PATH="ckpts/${RUN_NAME}"
VISUALIZATION_PATH="visualization/${RUN_NAME}"


#augmented used with tr+rot and rot models.
Data_file="datasets/train/af3_1022P_1022R_aug.csv" #csv file for the data names and the split 
Data_path="datasets/train/af3_1022P_1022R_aug" #dir to the stored dill files

# Data_file="datasets/train/af3_1022P_1022R.csv" #csv file for the data names and the split
# Data_path="datasets/train/af3_1022P_1022R"  #dir to the stored dill files

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
    --project "STRAND Tuning" \
    --visualize_n_val_graphs 0 \
    --visualization_path $VISUALIZATION_PATH \
    --tr_weight 0.5 \
    --rot_weight 0.5\
    --knn_size 30\
    --patience 250\
    --nv 4 --ns 16 --max_radius 5 \
    --recache \
    --data_path $Data_path\
    --data_file $Data_file\
    --translation True\
    --rotation True\
    --torsion False\
    --recache \
    #--checkpoint_path $SAVE_PATH \
    #--debug True # load small dataset
    #--entity coarse-graining-mit \

# if you accidentally screw up and the model crashes
# you can restore training (including optimizer)
# by uncommenting --checkpoint_path