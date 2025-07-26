import random
import pickle
import os
from scipy.spatial import KDTree
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import argparse



def process_single_file(dill_file, dill_data_path, save_path, distance_cutoff=7):
    """Process a single dill file to find contact chains"""
    try:
        data_path = os.path.join(dill_data_path, dill_file)
        find_contact_chains_linear(data_path, distance_cutoff, save_path)
    except Exception as e:
        print(f"Error processing {dill_file}: {str(e)}")

def find_contact_chains_linear(data_path_dill, distance_cutoff=7, save_path=None):
    # Load data
    with open(data_path_dill, 'rb') as f:
        data = pickle.load(f)
    
    # Filter RNA and protein atoms
    rna_residues = data[(data['type'] == 'rna') & (data['atom_name'] == 'P')].reset_index()
    protein_residues = data[(data['type'] == 'protein') & (data['atom_name'] == 'CA')].reset_index()
    
    # Build KD-tree for protein coordinates
    protein_coords = protein_residues[['x', 'y', 'z']].to_numpy()
    protein_tree = KDTree(protein_coords)
    
    # RNA coordinates
    rna_coords = rna_residues[['x', 'y', 'z']].to_numpy()
    
    # Find pairs within the distance cutoff
    indices = protein_tree.query_ball_point(rna_coords, r=distance_cutoff)
    
    # Collect close contacts
    close_contacts = set()
    for rna_index, protein_indices in enumerate(indices):
        if protein_indices:
            rna_chain = rna_residues.loc[rna_index, 'chain']
            for protein_index in protein_indices:
                protein_chain = protein_residues.loc[protein_index, 'chain']
                
                # Ensure unique contacts
                contact = (protein_chain, rna_chain)
                if contact not in close_contacts:
                    close_contacts.add(contact)
                    
                    # # Separate RNA and protein explicitly #dump based on res length
                    rna_chain_data = data[(data['chain'] == rna_chain) & (data['type'] == 'rna') & (data['atom_name'] == 'P')]  #
                    protein_chain_data = data[(data['chain'] == protein_chain) & (data['type'] == 'protein') & (data['atom_name'] == 'CA')]
                    
                    # #dump based on atom length
                    # rna_chain_data = data[(data['chain'] == rna_chain) & (data['type'] == 'rna') ]
                    # protein_chain_data = data[(data['chain'] == protein_chain) & (data['type'] == 'protein')] 


                    # # # Separate RNA and protein explicitly #dump based on res length
                    # rna_chain_data = data[(data['chain'] == rna_chain) & (data['type'] == 'rna') ]
                    # protein_chain_data = data[(data['chain'] == protein_chain) & (data['type'] == 'protein') & (data['atom_name'] == 'CA')]

                    
                    # Skip if lengths are invalid
                    if not (1 <= len(rna_chain_data) <= 1022 and 1 <= len(protein_chain_data) <= 1022):
                        continue
                    
                    # Generate file name
                    pdb_id_name = os.path.basename(data_path_dill).split('.')[0]
                    file_name = f"{pdb_id_name}_{protein_chain}_{rna_chain}.dill"
                    
                    # Save RNA-protein pair
                    if save_path:
                        output_file = os.path.join(save_path, file_name)
                        with open(output_file, 'wb') as f:
                            pickle.dump(pd.concat([rna_chain_data, protein_chain_data]), f)
       
def main(args):
    # Set your paths

    dill_data_path = args.dir_path
    save_path = args.save_path    
    dill_files=os.listdir(dill_data_path)
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    random.shuffle(dill_files)
    
    # Set up multiprocessing
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    print(f"Found {mp.cpu_count()} CPUs")   
    print(f"Starting processing with {num_processes} processes")
    
    # Create partial function with fixed arguments
    process_file = partial(process_single_file, 
                         dill_data_path=dill_data_path,
                         save_path=save_path,
                         distance_cutoff=7)
    
    # Create pool and process files
    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_file, dill_files) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create file pairs")
    parser.add_argument("--dir_path", type=str, required=True, help="Directory containing the files to be paired.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the paired files.")
    args = parser.parse_args()
    main(args)