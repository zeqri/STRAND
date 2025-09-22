# STRAND  
RNA-Protein Complex Refinement via Diffusion

# Installation 

clone the repo

```
git clone https://github.com/zeqri/STRAND.git

```


```
conda create -n STRAND python=3.9.18
conda activate STRAND
pip install -r requirements.txt
```


# STRAND   
RNA-Protein Complex Refinement via Diffusion

# Installation   

Clone the repo:
``` 
git clone https://github.com/zeqri/STRAND.git  
```   

Create and activate conda environment:
``` 
conda create -n STRAND python=3.9.18 
conda activate STRAND 
pip install -r requirements.txt 
```   

# Replicating Paper Results

## 📁 Dataset Setup

Download the preprocessed structures used in our experiments:

👉 **[Download Structures from Google Drive](https://drive.google.com/drive/folders/1y34yiHQTAeMjCN49Crmi_tf4HwRlRbod?usp=sharing)**

1. Download the `test_data.zip` file 
2. Place it in the `datasets` directory
3. Extract the archive:

```bash
unzip datasets/test_data.zip -d datasets/
```

## 🚀 Running Experiments

Choose your inference mode based on whether you want to use the confidence model:

### Manual Selection (Without Confidence Model)

Run structure refinement with manual selection on any of the three benchmark datasets:

```bash
# RNA-Pro dataset
sh src/inference_manual.sh rnapro

# Non-X-ray dataset  
sh src/inference_manual.sh nonxray

# X-ray dataset
sh src/inference_manual.sh xray
```

### Automated Selection (With Confidence Model)

Run structure refinement using the confidence model for automated structure selection:

```bash
# RNA-Pro dataset
sh src/inference_conf.sh rnapro

# Non-X-ray dataset
sh src/inference_conf.sh nonxray

# X-ray dataset
sh src/inference_conf.sh xray
```

## 📊 Results

Refined structures and evaluation metrics will be saved in the `results/` directory, organized by dataset and method used.


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

After obtaining an optimised SCORE MODEL, use it to generate samples via:

```
sh src/generate_samples.sh
```



**Confidence model:**

Use the generated samples to train the confidence model and run: 

```
sh src/train_confidence.sh

```


# Inference 

Store the structures to be refined as dill files using `src/data/preprocessing/cache_data.py`.

Specify the path of the stored data set to be refined and it's corrosponding csv file in the variables `Data_path` and `Data_path` respectively in the file `src/train_confidence.sh`.


Set `--run_inference_without_confidence_model` to be True to run the inference without the confidence model. 

Run the inferecne porcess via: 

```
sh src/inference.sh
```


Set `--run_inference_without_confidence_model` to be False to run the inference without the confidence model. 

```
sh src/inference.sh

```


After running the inference visualization directories are created containing the generated samples. Defualt path is `visualization/STRAND`

To assess how well the refined samples are, Downdload the Ground Truth files that were refined from the PDB as .pdb and store them in`datasets/gt_dir`


**Manual Selection results:** 

To display manual selection results run:



```
python src/visualize_inf_manual.py  --gt_path datasets/gt_dir --samples_path visualization/STRAND

```


**Selection via confidence model:** 


To display the confidence model selection results run:



```
python src/visualize_inf_conf.py  --gt_path datasets/gt_dir --samples_path visualization/STRAND

```





