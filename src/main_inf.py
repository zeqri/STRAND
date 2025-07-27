import os
import sys
import yaml
import random
import resource
from collections import defaultdict
import copy
import numpy as np
import torch
from tqdm import tqdm
import pickle
import time

from args import parse_args
from data import load_data, get_data,BindingDataset
from model import load_model, to_cuda
from utils import printt, print_res, log, get_unixtime
from train import train, evaluate, evaluate_pose
from filtering.dataset import get_confidence_loader
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from geom_utils import set_time
# from helpers import WandbLogger, TensorboardLogger
from sample import sample  
from evaluation.compute_rmsd import evaluate_all_rmsds
import pickle 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def evaluate_confidence(model,loader,args):
    
    all_confidences = []
    all_confidences_with_name ={}
    

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_labels = []
    all_pred = []
    all_loss = []
    print("loader len: ",len(loader))
    for data in tqdm(loader, total=len(loader)):
        #data, rmsd = batch
        # move to CUDA

        
        if args.num_gpu == 1 and torch.cuda.is_available():
            data = data.cuda()
            set_time(data, 0, 0, 0, batch_size=args.batch_size, device=device)
        else: 
            set_time(data, 0, 0, 0, batch_size=args.batch_size, device=device)
        try:
            with torch.no_grad():
                pred = model(data)
            #print("prediction",pred)
            all_pred.append(pred.detach().cpu())

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    all_pred = torch.cat(all_pred).tolist() # TODO -> maybe list inside


    return all_pred


def main(args=None):
    printt("Starting Inference")

    if args is None:
        args = parse_args()
    print(args)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    torch.hub.set_dir(args.torchhub_path)

    start_time = time.time()
    # load raw data
    data = load_data(args)
    data_params = data.data_params
    printt("finished loading raw data")

    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if args.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    
    # test mode: load up all replicates from checkpoint directory
    # and evaluate by sampling from reverse diffusion process
    if args.mode == "test":
        set_seed(args.seed)
        printt("running inference")
        fold = 0
        # load and convert data to DataLoaders
        loaders = get_data(data, fold, args, for_reverse_diffusion=True)
        # print(loaders["test"].data)
        
        printt("finished creating data splits")
        # get model and load checkpoint, if relevant
        model = load_model(args, data_params, fold,confidence_mode=False)
        model_confidence = load_model(args, data_params, fold,confidence_mode=True)
        
        if torch.cuda.is_available(): 
            model = to_cuda(model, args)
            model_confidence = to_cuda(model_confidence, args)
        
        if args.run_inference_without_confidence_model:
            printt('Running sequentially without confidence model')
            full_list = [loaders["val"]]
            complex_rmsd_lt5 = []
            complex_rmsd_lt2 = []
            time_to_load_data = time.time() - start_time
            print(f'time_to_load_data: {time_to_load_data}')
            start_time = time.time()
            for i in tqdm(range(1)):
                samples_list = sample(
                    loaders["val"], 
                    model, 
                    args, 
                    visualize_first_n_samples=args.visualize_n_val_graphs, 
                    visualization_dir=args.visualization_path,)
        
                full_list.append(samples_list)
                meter = evaluate_all_rmsds(loaders["val"], samples_list)
                ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized = meter.summarize(verbose=True)
                complex_rmsd_lt5.append(complex_rmsd_summarized['lt5'])
                complex_rmsd_lt2.append(complex_rmsd_summarized['lt2'])
                printt(f'Finished {i}-th sweep over the data')

            end_time = time.time()
            print(f'Total time spent processing 5 times: {end_time-start_time}')
            print(f'time_to_load_data: {time_to_load_data}')

            complex_rmsd_lt5 = np.array(complex_rmsd_lt5)
            complex_rmsd_lt2 = np.array(complex_rmsd_lt2)
            print(f'Average CRMSD < 5: {complex_rmsd_lt5.mean()}')
            print(f'Average CRMSD < 2: {complex_rmsd_lt2.mean()}')
            dump_predictions(args,full_list)
            printt("Dumped data!!")
            return

        # run reverse diffusion process
        print(f'args.temp_sampling: {args.temp_sampling}')
        loaders,results = generate_loaders(loaders["val"],args) #TODO adapt sample size
        best_index_dict = {}      
        for i, loader in tqdm(enumerate(loaders), total=len(loaders)):
            printt(f'loader {i} len: {len(loader)}')
            
            for batch in loader:
                original = copy.deepcopy(batch)
                  
            samples_list = sample(loader, model, args, visualize_first_n_samples=args.visualize_n_val_graphs, visualization_dir=args.visualization_path)
            samples_list.append(original) 
            
            assert len(samples_list)==41
            samples_loader = DataLoader(samples_list, batch_size=args.batch_size)
            for data in samples_loader:
                name = data.name
                break
            pred_list = evaluate_confidence(model_confidence, samples_loader, args)
            sorted_pairs = sorted(zip(samples_list, pred_list), key=lambda x: x[1])
            for graph, pred in sorted_pairs:
                printt(f"Graph name: {graph.name}, Prediction: {pred:.4f}")
            results[i] += sorted_pairs
            sorted_samples = [pair[0] for pair in sorted_pairs]
            sorted_indexes = [samples_list.index(sample) for sample in sorted_samples]
            printt(f"Sorted sample indices for complex {name}: {sorted_indexes}")
            printt("Finished Complex!")
            best_index_dict[name[0]] = sorted_indexes 

        best_index_path=f"{args.visualization_path}/{args.run_name}_sorted_index_dict.pkl"
        with open(best_index_path, 'wb') as f:
             pickle.dump(best_index_dict, f)

        printt(f'Finished run {args.run_name}')
        print(f'temp sampling, temp_psi, temp_sigma_data_tr, temp_sigma_data_rot: {args.temp_sampling, args.temp_psi, args.temp_sigma_data_tr, args.temp_sigma_data_rot}')
        print(f'filtering_model_path: {args.filtering_model_path}')
        end_time = time.time()
        print(f'Total time spent: {end_time-start_time}')
        meter = evaluate_all_predictions(results)
        #reverse_diffusion_metrics = evaluate_predictions(results)
        #printt(reverse_diffusion_metrics)
        
        dump_predictions(args,results)
        printt(f"Dumped data!! in {args.prediction_storage}")
        
        # log(test_scores, args.log_file, reduction=False)
        # end of all folds ========

def evaluate_all_predictions(results):
    ground_truth = [res[0][0] for res in results]
    best_pred = [res[1][0] for res in results]
    meter = evaluate_all_rmsds(ground_truth,best_pred)
    _ = meter.summarize()
    return meter


def evaluate_predictions(results):
    ground_truth = [res[0][0] for res in results]
    best_pred = [res[1][0] for res in results]
    eval_result = evaluate_pose(ground_truth,best_pred)
    rmsds = np.array(eval_result["rmsd"])
    reverse_diffusion_metrics = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
                                    'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
                                    'rmsds_lt10': (100 * (rmsds < 10).sum() / len(rmsds)),
                                    'rmsds_mean': rmsds.mean(),
                                    'rmsds_median': np.median(rmsds)}
    return reverse_diffusion_metrics

def dump_predictions(args,results):
    with open(args.prediction_storage, 'wb') as f:
        pickle.dump(results, f)

def load_predictions(args):
    with open(args.prediction_storage, 'rb') as f:
        results = pickle.load(f)
    return results


def generate_loaders(loader,args):
    result = []
    data = loader.data
    ground_truth = []
    for d in data:
        #element = BindingDataset(args, {}, apply_transform=False)
        data_list = []
  
        for i in range(args.num_samples):
            data_list.append(copy.deepcopy(d))

        if args.mirror_ligand:
            #printt('Mirroring half of the complexes')
            for i in range(0, args.num_samples//2):
                e = data_list[i]["graph"]

                data = HeteroData()
                data["name"] = e["name"]

                data["receptor"].pos = e["ligand"].pos
                data["receptor"].x = e["ligand"].x

                data["ligand"].pos = e["receptor"].pos
                data["ligand"].x = e["receptor"].x

                data["receptor", "contact", "receptor"].edge_index = e["ligand", "contact", "ligand"].edge_index
                data["ligand", "contact", "ligand"].edge_index = e["receptor", "contact", "receptor"].edge_index

                # center receptor at origin
                center = data["receptor"].pos.mean(dim=0, keepdim=True)
                for key in ["receptor", "ligand"]:
                    data[key].pos = data[key].pos - center
                data.center = center  # save old center
                data["mirrored"] = True

                data_list[i]["graph"] = data

        element = BindingDataset(args, data_list, apply_transform=False)
        
        result.append(element)
    
    for element in loader:
        ground_truth.append([(element,float("inf"))])
    return result,ground_truth



if __name__ == "__main__":
    main()