import pickle
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation as R
import pandas as pd
import random
import argparse
import shutil


def set_seed(seed):
    """
    Sets the seed for reproducibility.
    :param seed: Integer seed value.
    """
    random.seed(seed)  # For Python random
    np.random.seed(seed)  # For numpy
    torch.manual_seed(seed)  # For PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_pdb(file, i, save_path, reflection_vector):
    """
    Augments a PDB file by applying transformations.

    :param file: Path to the input PDB file.
    :param i: Index of the augmented file.
    :param save_path: Directory to save the augmented file.
    :param reflection_vector: Vector for reflection transformation.
    """
    # Read original PDB
    with open(file, "rb") as f:
        data = pickle.load(f)
    
    file_name = file.split("/")[-1].split(".")[0]
    new_path = os.path.join(save_path, f"{file_name}_mod_{i}.dill")

    # Apply random rotation and translation
    cords = torch.tensor(data[['x', 'y', 'z']].values)
    # center = torch.mean(cords, dim=0, keepdim=True).float()
    # random_rotation = torch.from_numpy(R.random().as_matrix()).float()
    tr_update = torch.normal(0, 30, size=(1, 3))

    # Reflection augmentation
    if i==0:    
        with open(new_path, "wb") as f: 
            pickle.dump(data, f)    
    else:
        # Apply random rotation
        cords = cords * reflection_vector
        pos = cords + tr_update
        data[['x', 'y', 'z']] = pos.numpy()
        with open(new_path, "wb") as f:
            pickle.dump(data, f)    
         


def main(args):
    set_seed(args.seed)

    # Parse all data
    data = pd.read_csv(args.csv_path)

    # Keep a separate copy of full data
    data_train = data[data['split'] == 'train']
    file_list = data_train['path'].tolist()

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Augment only train files
    for file in file_list:
        file_path = os.path.join(args.dir_path, file)
        for i in range(args.num_augmentations):
            reflection_vector = (torch.randint(0, 2, (3,)) * 2 - 1).float()
            augment_pdb(file_path, i, args.save_path, reflection_vector)

    # Now create a new dataframe
    file_list = os.listdir(args.save_path)
    df = pd.DataFrame(file_list, columns=['path'])
    df['split'] = 'train' 

    for split in ['val', 'test']:
        data_split = data[data['split'] == split]
        for file in data_split['path'].tolist():
            src_path = os.path.join(args.dir_path, file)
            dst_path = os.path.join(args.save_path, file)
            shutil.copy(src_path, dst_path)



    # Add val and test splits from original full data
    df_val = data[data['split'] == 'val'].copy()
    df_test = data[data['split'] == 'test'].copy()

    # (df_val and df_test already have 'split' set correctly)

    df = pd.concat([df, df_val, df_test], ignore_index=True)

    # Save the new CSV
    df.to_csv(args.output_csv, index=False)

    print(f"New CSV saved to {args.output_csv}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment PDB files with transformations.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--dir_path", type=str, required=True, help="Directory containing original dill files.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save augmented files.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with file information.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the new CSV file.")
    parser.add_argument("--num_augmentations", type=int, default=3, help="Number of augmentations per file.")

    args = parser.parse_args()
    main(args)

