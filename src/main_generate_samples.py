import os
import shutil
import random
import resource

import time

import numpy as np
import torch
import torch.nn as nn

from evaluation.compute_rmsd import rigid_transform_Kabsch_3D
from geom_utils.geometry import kabsch_torch
from utils import compute_rmsd


from args import parse_args
from data import load_data, get_data
from model import load_model, to_cuda
from utils import printt
from sample import sample
from tqdm import tqdm



def update_complex_rmsd( ligand_coors_pred, ligand_coors_true, receptor_coors):
        complex_coors_pred = torch.concatenate((ligand_coors_pred, receptor_coors), axis=0)
        complex_coors_true = torch.concatenate((ligand_coors_true, receptor_coors), axis=0)
        R,t = kabsch_torch(complex_coors_pred.T, complex_coors_true.T)
        complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T
        complex_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)

        return complex_rmsd

def evaluate_pose_NEW(data_list, samples_list):
    """
        Evaluate sampled pose vs. ground truth
    """
    complex_rmsd_list = []
    all_rmsds = []
    rmsds_with_name = {}
    complex_rmsds_with_name = {}

    assert len(data_list) == len(samples_list)
    for true_graph, pred_graph in zip(data_list, samples_list):
        rec_xyz = true_graph["receptor"].pos    
        true_xyz = true_graph["ligand"].pos
        pred_xyz = pred_graph["ligand"].pos
        if true_xyz.shape != pred_xyz.shape:
            print(true_graph["name"], pred_graph["name"])
        assert true_xyz.shape == pred_xyz.shape
        rmsd = compute_rmsd(true_xyz, pred_xyz)
        complex_rmsd= update_complex_rmsd(pred_xyz, true_xyz, rec_xyz)
        complex_rmsd_list.append(complex_rmsd)  
        all_rmsds.append(rmsd)
        rmsds_with_name[true_graph["name"]] = rmsd 
        complex_rmsds_with_name[true_graph["name"]] = complex_rmsd

    scores = {
        "rmsd": all_rmsds,
        "complex_rmsd": complex_rmsd_list,
        "rmsds_with_name": rmsds_with_name, 
        "complex_rmsds_with_name": complex_rmsds_with_name, 
    }

    return scores


DATA_CACHE_VERSION = "v1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args=None):
    
    if args is None:
        args = parse_args()
    print(args)
    if torch.cuda.is_available():   
        torch.cuda.set_device(args.gpu)
        torch.hub.set_dir(args.torchhub_path)
    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if args.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    fold = 0

    #### set up fold experiment
    set_seed(args.seed)

    # load data params
    data_params = {}
    data_params["num_residues"] = 23
    data = load_data(args, split="train", batch=0, verbose=False)

    print("Loaded first batch")

    # get model and load checkpoint, if relevant
    model = load_model(args, data_params, fold, load_best=True) # load last_model to continue training
    if torch.cuda.is_available():
        model=to_cuda()
    printt("finished loading model")
    split_samples ={
            "train": 1,    
            "val": 1,   
            "test": 1
 
        }
    # Log number of parameters
    numel = sum([p.numel() for p in model.parameters()])
    printt('Model with', numel, 'parameters')
    # Get cache directory

    for split in ( "train", "val" , "test"):
    # for split in ( "val" , "test"):
        print(f"Inference for {split} split!")

        # n_batches = DIPSLoader.get_n_batches()[split]
        n_batches = split_samples[split]    
        batch_indexes = list(range(n_batches))
        # get_random_indexes_ignore_seed(batch_indexes, args.seed)
        random.shuffle(batch_indexes)

        for batch_index in tqdm(batch_indexes):
            # Get current directory
            directory = f"{args.samples_directory}/{split}/batch-{batch_index}"
# 
            # If directory exists and is not empty, continue
            if os.path.exists(directory):
                print(f"batch {batch_index} is already generated. Continue!")
                continue

            # Load batch
            data = load_data(args, split=split, batch=batch_index, verbose=False)
            batch = get_data(data, fold, args, for_reverse_diffusion=True)[split]
            if len(batch) == 0:
                print("Zero batch!")
                continue
            # If not, create directory and continue
            os.makedirs(directory, exist_ok=True)
            # run reverse diffusion process
            samples_multiple_iterations = []
            rmsd_multiple_iterations = []
            crmsd_multiple_iterations = []  
            try:
                for i in range(args.generate_n_predictions):
                    start_time = time.time()    
                    print(f"Generating sample {i} for batch {batch_index}.")
                    iteration = sample(batch, model, args) 
                    scores = evaluate_pose_NEW(batch, iteration)
                    rmsd= scores["rmsd"]    
                    crmsd= scores["complex_rmsd"] 
                    samples_multiple_iterations.append(iteration)
                    rmsd_multiple_iterations.append(rmsd)
                    crmsd_multiple_iterations.append(crmsd)
                    end_time = start_time - time.time()
                    #print time in miniutes  
                    print(f"Time taken: {int(end_time // 60)} minutes {end_time % 60:.2f} seconds")
                   
            except Exception as e:
                printt("RuntimeError. ")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                shutil.rmtree(directory)
                continue

            # For some reason, there were some samples without complex_t
            if "complex_t" not in samples_multiple_iterations[0][0]:
                print(f"No complex_t! Batch: {batch_index}.")

    
            serialize_new(samples_multiple_iterations, rmsd_multiple_iterations,crmsd_multiple_iterations, directory=directory)
              

def get_random_indexes_ignore_seed(batches, current_seed):
    batch_indexes = list(range(len(batches)))

    # Randomize batch indexes
    random.seed(None)
    random.shuffle(batch_indexes)
    random.seed(current_seed)
    return batch_indexes


def split_list_into_batches(data, batch_size):
    number_of_batches = len(data) // batch_size + int(len(data) % batch_size > 0)
    batches = [data[i: i * batch_size] for i in range(number_of_batches)]
    return batches



def serialize_new(samples_multiple_iterations, rmsd_multiple_iterations, crmsd_multiple_iterations, directory="."):
    torch.save(samples_multiple_iterations[0], f"{directory}/first_iteration.pkl")
    ligand_positions = [[graph["ligand"].pos for graph in graphs] for graphs in samples_multiple_iterations]
    torch.save(ligand_positions, f"{directory}/ligand_positions.pkl")
    torch.save(rmsd_multiple_iterations, f"{directory}/rmsds.pkl")
    torch.save(crmsd_multiple_iterations, f"{directory}/crmsds.pkl")




if __name__ == "__main__":
    main()

