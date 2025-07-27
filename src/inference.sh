NUM_FOLDS=1  # number of seeds to try, default 5
SEED=0  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=1
BATCH_SIZE=1  # split across all GPUs
NUM_SAMPLES=40

NAME="rna_inf"  # change to name of config file
RUN_NAME="STRAND"
CONFIG="config/${NAME}.yaml"

SAVE_PATH="ckpts/${RUN_NAME}"
VISUALIZATION_PATH="visualization/${RUN_NAME}"
STORAGE_PATH="storage/${RUN_NAME}.pkl"

FILTERING_PATH="models/conf_tr_max_0.1_rmsd_regression_30/fold_0" #trained models 
SCORE_PATH="models/small_noise_aug_tr_max_0.1/fold_0" #trained models

Data_path="datasets/test_data/rnapor/prediction"
Data_file="datasets/test_data/rnapor/rnapro.csv"


echo SCORE_MODEL_PATH: $SCORE_PATH
echo CONFIDENCE_MODEL_PATH: $SCORE_PATH
echo SAVE_PATH: $SAVE_PATH

python src/main_inf.py \
    --mode "test" \
    --config_file $CONFIG \
    --run_name $RUN_NAME \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --num_folds $NUM_FOLDS \
    --num_gpu $NUM_GPU \
    --gpu $CUDA --seed $SEED \
    --logger "wandb" \
    --project "DiffDock Tuning" \
    --visualize_n_val_graphs 100 \
    --visualization_path $VISUALIZATION_PATH \
    --filtering_model_path $FILTERING_PATH \
    --score_model_path $SCORE_PATH \
    --num_samples $NUM_SAMPLES \
    --prediction_storage $STORAGE_PATH \
    --knn_size 30\
    --data_path $Data_path\
    --data_file $Data_file\
    --run_inference_without_confidence_model #flag to use confidence model or not
    #--entity coarse-graining-mit \
    #--debug True # load small dataset

