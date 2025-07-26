# STRAND  
RNA-Protein Complex Refinement via Diffusion

**Note:** This repository is currently under development.  
The complete code will be made publicly available upon the acceptance or publication of the associated research paper. 


# Data Preprocessing

Download PDB files containing RNA-protein complexe before the cutoff date 30.Sept.2021 

All the data must be stored as dill files, to do so run:

```
python src/data/preprocessing/cache_data.py --dir_path dir_path_containing_pdb_file --save_path datasets/train
```

Strand tr+rot utalized data augmentation during training, to augment the data run:

```
sh src/data/preprocessing/data_aug.sh

```


# Training 

**SCORE MODEL:**

Download PDB files containing RNA-protein complexe before the cutoff date 30.Sept.2021 and store them into `datasets/pdb_files`. 

All the data must be stored as dill files, to do so run:

```
python src/data/preprocessing/cache_data.py --dir_path datasets/pdb_files --save_path datasets/train/af3_1022P_1022R 
```


Strand tr+rot utalized data augmentation during training, to augment the data run:

```
sh src/data/preprocessing/data_aug.sh

```


To start training the STRAND tr+rot run: 

```
sh src/train.sh

```

**Generate samples:**

After obtaining an optimised SCORE MODEL, use it to generate sampoles via:

```
sh src/generate_samples.sh
```



**Confidence model:**

Use the generated samples to train the confidence model and run: 

```
sh src/train_confidence.sh

```



# Inference 

Store the structurs to be refined as dill files using `src/data/preprocessing/cache_data.py`.

Specify the path of the stored data set to be refined and it's corrosponding csv file in the variables `Data_path` and `Data_path` respectively in the file `src/train_confidence.sh`.



**Manual Selection results:**


Set `--run_inference_without_confidence_model` to be True to run the inference without the confidence model. 

Run the inferecne porcess via: 

```
sh src/inference.sh
```



**Selection via confidence model:** 


Set `--run_inference_without_confidence_model` to be False to run the inference without the confidence model. 

```
sh src/inference.sh

```