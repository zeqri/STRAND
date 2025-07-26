import os
import csv
import yaml
import glob
import itertools
from collections import defaultdict

import time
from datetime import datetime

from fpdf import FPDF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

# -------- general

def load_csv(fp, split=None, batch=None):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f)
        for line in reader:
            # Skip not specified batches & splits
            if split is not None and line["split"] != split:
                continue
            if batch is not None and int(line["batch"]) != int(batch):
                continue
            data.append(line)
    return data


def log(item, fp, reduction=True):
    # pre-process item
    item_new = {}
    for key, val in item.items():
        if type(val) is list and reduction:
            key_std = f"{key}_std"
            item_new[key] = float(np.mean(val))
            item_new[key_std] = float(np.std(val))
        else:
            if torch.is_tensor(val):
                item[key] = val.tolist()
            item_new[key] = val
    # initialization: write keys
    if not os.path.exists(fp):
        with open(fp, "w+") as f:
            f.write("")
    # append values
    with open(fp, "a") as f:
        yaml.dump(item_new, f)
        f.write(os.linesep)


def chain(iterable, as_set=True):
    if as_set:
        return sorted(set(itertools.chain.from_iterable(iterable)))
    else:
        return list(itertools.chain.from_iterable(iterable))


def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def get_unixtime():
    timestamp = str(int(time.time()))
    return timestamp


def printt(*args, **kwargs):
    print(get_timestamp(), *args, **kwargs)


def print_res(scores):
    """
        @param (dict) scores key -> score(s)
    """
    for key, val in scores.items():
        if type(val) is list:
            print_str = f"{np.mean(val):.3f} +/- {np.std(val):.3f}"
            print_str = print_str + f" ({len(val)})"
        else:
            print_str = f"{val:.3f}"
        print(f"{key}\t{print_str}")


def get_model_path(fold_dir):
    # load last model saved (we only save if improvement in validation performance)
    # convoluted code says "sort by epoch, then batch"
    # new code says "sort by rmsd, take the lowest"
    paths = []
    for path in glob.glob(f"{fold_dir}/*.pth"):
        if "last" not in path:
            paths.append(path)
    models = sorted(paths, key=lambda s:float(s.split("/")[-1].split("_")[4]))
                    #key=lambda s:(int(s.split("/")[-1].split("_")[3]),
                    #              int(s.split("/")[-1].split("_")[2])))
    if len(models) == 0:
        print(f"no models found at {fold_dir}")
        return
    checkpoint = models[0]
    return checkpoint

def select_model(fold_dir,confidence_mode):
    paths = []
    for path in glob.glob(f"{fold_dir}/*.pth"):
        if "last" not in path:
            paths.append(path)

    if confidence_mode:
        models = sorted(paths, key=lambda s:-float(s.split("/")[-1].split("_")[-1][:-4]))
    else:
        models = sorted(paths, key=lambda s:float(s.split("/")[-1].split("_")[4]))

    if len(models) == 0:
        print(f"no models found at {fold_dir}")
        return
    checkpoint = models[0]
    return checkpoint



def get_optimizer(model, args, load_best=True, confidence_mode=False):
    """
        Initialize optimizer and load if applicable
    """
    optimizer = Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)
    #optimizer = SGD(model.parameters(),
    #                lr=args.lr,
    #                momentum=0.5,
    #                weight_decay=args.weight_decay)
    # SGD is awful for my models. don't use it.
    ## load optimizer state
    fold_dir = args.fold_dir
    if args.checkpoint_path is not None:
        if load_best:
            checkpoint = select_model(fold_dir, confidence_mode=confidence_mode)
        else:
            checkpoint = os.path.join(fold_dir, "model_last.pth")

        if checkpoint is not None:
            #start_epoch = int(checkpoint.split("/")[-1].split("_")[3])
            start_epoch = 0
            with torch.no_grad():
                optimizer.load_state_dict(torch.load(checkpoint,
                    map_location="cpu")["optimizer"])
            printt("Finished loading optimizer")
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    return start_epoch, optimizer


def init(model):
    """
        Wrapper around Xavier normal initialization
        Apparently this needs to be called in __init__ lol
    """
    for name, param in model.named_parameters():
        # NOTE must name parameter "bert"
        if "bert" in name:
            continue
        # bias terms
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        # weight terms
        else:
            nn.init.xavier_normal_(param)

# -------- metrics

def compute_rmsd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = ((x-y)**2).sum(-1)
    dist = dist / len(dist)  # normalize
    dist = dist.sum().sqrt()
    return dist



def create_rmsd_report(complex_rmsd_values, min_rmsd_values, output_path="rmsd_report.pdf"):
    """
    Create a PDF report containing RMSD statistics.
    
    Parameters:
    complex_rmsd_values (list): List of complex RMSD values
    min_rmsd_values (list): List of minimum RMSD values
    plot_label (str): Label for the plot/analysis
    output_path (str): Path where to save the PDF
    """
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'RMSD Analysis Report', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    # Create PDF object
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add title and date
    pdf.set_font('Arial', 'B', 12)
    # pdf.cell(0, 10, f"Analysis Results - {plot_label}", 0, 1, 'L')
    pdf.cell(0, 10, f"Analysis Results", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    pdf.ln(10)
    
    # Basic Statistics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Basic Statistics", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Calculate statistics
    count = sum(min_rmsd < complex_rmsd for min_rmsd, complex_rmsd in zip(min_rmsd_values, complex_rmsd_values))
    total_complex = len(complex_rmsd_values)
    overall_percentage_change = ((np.mean(complex_rmsd_values) - np.mean(min_rmsd_values)) / np.mean(complex_rmsd_values)) * 100
    
    statistics = [
        f"Number of min RMSD less than complex RMSD: {count}",
        f"Number of total complex calculated: {total_complex}",
        f"Overall Percentage Change in mean value: {overall_percentage_change:.2f}%",
        f"Mean complex RMSD: {np.mean(complex_rmsd_values):.2f}",
        f"Median complex RMSD: {np.median(complex_rmsd_values):.2f}",
        f"Mean strand RMSD: {np.mean(min_rmsd_values):.2f}",
        f"Median strand RMSD: {np.median(min_rmsd_values):.2f}"
    ]
    
    for stat in statistics:
        pdf.cell(0, 8, stat, 0, 1, 'L')
    pdf.ln(10)
    
    # Complex RMSD Analysis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Complex RMSD Analysis", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    below_10 = [value for value in complex_rmsd_values if value < 10]
    below_5 = [value for value in complex_rmsd_values if value < 5]
    below_2 = [value for value in complex_rmsd_values if value < 2]
    
    complex_stats = [
        f"Percentage below 10Å: {(len(below_10) / len(complex_rmsd_values)) * 100:.2f}%",
        f"Percentage below 5Å: {(len(below_5) / len(complex_rmsd_values)) * 100:.2f}%",
        f"Percentage below 2Å: {(len(below_2) / len(complex_rmsd_values)) * 100:.2f}%"
    ]
    
    for stat in complex_stats:
        pdf.cell(0, 8, stat, 0, 1, 'L')
    pdf.ln(10)
    
    # Strand RMSD Analysis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Strand RMSD Analysis", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    below_10 = [value for value in min_rmsd_values if value < 10]
    below_5 = [value for value in min_rmsd_values if value < 5]
    below_2 = [value for value in min_rmsd_values if value < 2]
    
    strand_stats = [
        f"Percentage below 10Å: {(len(below_10) / len(min_rmsd_values)) * 100:.2f}%",
        f"Percentage below 5Å: {(len(below_5) / len(min_rmsd_values)) * 100:.2f}%",
        f"Percentage below 2Å: {(len(below_2) / len(min_rmsd_values)) * 100:.2f}%"
    ]
   
    for stat in strand_stats:
        pdf.cell(0, 8, stat, 0, 1, 'L')
    
    # Save the PDF
    pdf.output(output_path)
    print(f"Report saved to {output_path}")