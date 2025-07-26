import os 
import pickle 
import argparse
from preprocess_utils import extract_pdb_data


def main(args):     

    """
    Cache the data for faster processing when loading large amount of data
    
    """

    pdb_path=args.dir_path
    save_path=args.save_path 

    if not os.path.exists(save_path):
       os.makedirs(save_path)
    list_files=os.listdir(pdb_path) 
    
    for file in list_files: 
        file_path=os.path.join(pdb_path,file)
        data=extract_pdb_data(file_path) 
        pdb_id=file[:4]
        with open(os.path.join(save_path, f"{pdb_id}.dill"), "wb") as f:
              pickle.dump(data, f)  
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cache pdb files")
    parser.add_argument("--dir_path", type=str, required=True, help="Directory containing the files to be paired.")
    # parser.add_argument("--save_path", type=str, required=True, help="Directory to save the paired files.")
    parser.add_argument(
    "--save_path",
    type=str,
    default="datasets/train/af3_1022P_1022R",
    help="Directory to cache the pdb files"
)

    
    args = parser.parse_args()
    main(args)

