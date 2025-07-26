import os
from pathlib import Path
import argparse

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.compute_rmsd import   RMSDComputer
from helpers.helpers import extract_pdb_data ,align_dataframes 
from utils import create_rmsd_report


rmsd_computer = RMSDComputer() 


def process_and_plot_path(dir_path, plot_label, output_filename, gt_path,pdf_rep_path):
    gt_pdb_files = list(Path(dir_path).rglob("*gt.pdb"))
    pattern = re.compile(r".*-(\d+)\.pdb$")
    
    # Placeholder lists for RMSD values and labels
    complex_rmsd_values = []
    min_rmsd_values = []
    labels = []
    min_and_complex_rmsd = []
    min_rmsd_filenames = []

    # Iterate through ground truth files
    for file in gt_pdb_files:
        filename = file.parts[-2]
        gt_pdb = os.path.join(gt_path, filename + ".pdb")
        
        if not os.path.exists(gt_pdb):
            continue
        
        # Extract PDB data
        data_Gt = extract_pdb_data(gt_pdb)
        data_Gt = data_Gt[data_Gt["atom_name"].isin(["P", "CA"])] # Keep only CA and P atoms in the gt file.
        data_Af3 = extract_pdb_data(file) # refined structures contain only CA and P atoms.

        # Split into RNA and protein
        rna_Gt, protein_Gt = data_Gt[data_Gt['type'] == 'rna'], data_Gt[data_Gt['type'] == 'protein']
        rna_Af3, protein_Af3 = data_Af3[data_Af3['type'] == 'rna'], data_Af3[data_Af3['type'] == 'protein']

        # # Align RNA and protein data
        if len(rna_Gt) != len(rna_Af3):
            # continue
            rna_Gt, rna_Af3 = align_dataframes(rna_Gt, rna_Af3)
            
        if len(protein_Gt) != len(protein_Af3):
            # continue
            protein_Gt, protein_Af3 = align_dataframes(protein_Gt, protein_Af3)
        
        rna_res_seq_Af3 = "".join(rna_Af3["resname"].values)   
        rna_res_seq_Gt = "".join(rna_Gt["resname"].values)  
        assert rna_res_seq_Af3 == rna_res_seq_Gt, f"RNA sequence mismatch for {filename}" 
        protein_res_seq_Af3 = "".join(protein_Af3["resname"].values)    
        protein_res_seq_Gt = "".join(protein_Gt["resname"].values)
        assert protein_res_seq_Af3 == protein_res_seq_Gt, f"Protein sequence mismatch for {filename}"    

        # Recombine the aligned dataframes
        data_Gt = pd.concat([rna_Gt, protein_Gt])
        data_Af3 = pd.concat([rna_Af3, protein_Af3])
        
        # Extract coordinates
        receptor_coors = rna_Gt[['x', 'y', 'z']].values
        ligand_coors_true = protein_Gt[['x', 'y', 'z']].values
        ligand_coors_pred = protein_Af3[['x', 'y', 'z']].values
        receptor_coors_pred = rna_Af3[['x', 'y', 'z']].values

        # Compute RMSD
        complex_rmsd = rmsd_computer.update_complex_rmsd_strand(
            ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors
        )

        # Find the minimum RMSD from diffused structures
        min_rmsd = float('inf')
        min_rmsd_filename = None
        af3_dir_path = file.parent
        int_pdb_files = [p for p in af3_dir_path.iterdir() if pattern.match(p.name) and not p.name.endswith("-0.pdb") and not p.name.endswith("-41.pdb") ]

        for af3_diff_file in int_pdb_files:
            diffused_data = extract_pdb_data(af3_diff_file)
            ligand_coors_pred_diff = diffused_data[diffused_data['type'] == 'protein'][['x', 'y', 'z']].values

            # Compute RMSD for diffused structure
            complex_rmsd_diffused = rmsd_computer.update_complex_rmsd_strand(
                ligand_coors_pred_diff, ligand_coors_true, receptor_coors_pred, receptor_coors
            )
            
            if complex_rmsd_diffused < min_rmsd:
                min_rmsd = complex_rmsd_diffused
                min_rmsd_filename = af3_diff_file.name

        # Store results
        complex_rmsd_values.append(complex_rmsd)
        min_rmsd_values.append(min_rmsd)
        
        if filename.upper() == "8EDJ":    
            print(f"{plot_label} - min diffused crmsd 8EDJ: {min_rmsd}") 
            print(f"{plot_label} - AF3 pred crmsd 8EDJ: {complex_rmsd}") 
        if filename.upper() == "7VKL": 
            print(f"{plot_label} - min diffused crmsd 7VKL: {min_rmsd}") 
            print(f"{plot_label} - AF3 pred crmsd 7VKL: {complex_rmsd}") 
        
        min_and_complex_rmsd.append((min_rmsd, complex_rmsd))
        labels.append(filename)
        min_rmsd_filenames.append(min_rmsd_filename)

    
    create_rmsd_report(complex_rmsd_values, min_rmsd_values , pdf_rep_path)


    # Plot configuration
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{siunitx}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

    # Create plot
    x = np.arange(len(labels)) * 1 
    labels = [label.upper() for label in labels]
    width = 0.4

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, complex_rmsd_values, width, label='AF GT cRMSD', 
                    color="#1f77b4", edgecolor='black')
    rects2 = ax.bar(x + width/2, min_rmsd_values, width, label=f'{plot_label} Min Diffused cRMSD', 
                    color="#ff7f0e", edgecolor='black')

    ax.set_ylabel(r'Complex-RMSD (\si{\angstrom})')
    ax.set_xlabel('PDB Id')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    os.makedirs(args.pdf_path, exist_ok=True)
    os.makedirs(args.report_path, exist_ok=True)

    path_configs = [
        {
            'dir_path': args.dir_path,
            'plot_label': 'STRAND-tr+rot',
            'output_filename': f'{args.pdf_path}/manual_selection.pdf',
            'report_path': f'{args.report_path}/manual_selection.pdf'
        }
    ]

    for config in path_configs:
        print(f"\nProcessing {config['plot_label']}...")
        process_and_plot_path(
            config['dir_path'],
            config['plot_label'],
            config['output_filename'],
            args.gt_path,
            config['report_path']
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a specific STRAND result (manual selection)")

    parser.add_argument(
        "--gt_path",
        type=str,
        default="datasets/gt_dir",
        help="Path to ground truth residue directory"
    )

    parser.add_argument(
        "--samples_path",
        type=str,
        default="visualization/STRAND",
        help="Directory containing prediction data"
    )

    parser.add_argument(
        "--pdf_path",
        type=str,
        default="results/plots",
        help="Directory to save PDF plots (default: results/plots)"
    )

    parser.add_argument(
        "--report_path",
        type=str,
        default="results/reports",
        help="Directory to save report PDFs (default: results/reports)"
    )

    args = parser.parse_args()
    main(args)