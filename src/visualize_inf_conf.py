import pandas as pd

import os
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


from evaluation.compute_rmsd import   RMSDComputer
from data.preprocessing.preprocess_utils import extract_pdb_data ,align_dataframes  
from utils import create_rmsd_report


rmsd_computer = RMSDComputer() 

def process_and_plot_path(dir_path, plot_label, output_filename, gt_path,pdf_rep_path ,dict_file=None,n=5):
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
        data_Af3 = extract_pdb_data(file)

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
        pdb_id = file.parent.name + ".dill"  
        best_index = [str(x+1) for x in dict_file[pdb_id]][:n]  

        
        int_pdb_files = [p for p in af3_dir_path.iterdir()
             if pattern.match(p.name) 
             and not p.name.endswith("-0.pdb") 
             and pattern.match(p.name).group(1) in best_index]  
        assert len(int_pdb_files) == n, f"No diffused structures found for {filename}"
   
        

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
    rects1 = ax.bar(x - width/2, complex_rmsd_values, width, label='Model predictions cRMSD', 
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

    # Automatically find .pkl file in dir_path
    dict_path = next(
        (os.path.abspath(os.path.join(args.dir_path, f)) for f in os.listdir(args.dir_path) if f.endswith(".pkl")),
        None
    )

    if dict_path is None:
        raise FileNotFoundError(f"No .pkl file found in {args.dir_path}")

    with open(dict_path, 'rb') as f:
        dict_file = pickle.load(f)

    path_configs = [
        {
            'dir_path': args.dir_path,
            'plot_label': 'STRAND-tr+rot',
            'output_filename': f'{args.pdf_path}/top_1.pdf',
            'report_path': f'{args.report_path}/top_1.pdf',
            'n': 1,
        },
        {
            'dir_path': args.dir_path,
            'plot_label': 'STRAND-tr+rot',
            'output_filename': f'{args.pdf_path}/top_5.pdf',
            'report_path': f'{args.report_path}/top_5.pdf',
            'n': 5,
        },
        {
            'dir_path': args.dir_path,
            'plot_label': 'STRAND-tr+rot',
            'output_filename': f'{args.pdf_path}/top_10.pdf',
            'report_path': f'{args.report_path}/top_10.pdf',
            'n': 10,
        },
        {
            'dir_path': args.dir_path,
            'plot_label': 'STRAND-tr+rot',
            'output_filename': f'{args.pdf_path}/top_20.pdf',
            'report_path': f'{args.report_path}/top_20.pdf',
            'n': 20,
        },
    ]

    for config in path_configs:
        print(f"\nProcessing {config['plot_label']}...")
        process_and_plot_path(
            config['dir_path'],
            config['plot_label'],
            config['output_filename'],
            args.gt_path,
            config['report_path'],
            dict_file=dict_file,
            n=config['n']
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for STRAND")

    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth residue directory")
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
    parser.add_argument("--samples_path", type=str, required=True, help="Directory containing refined data and .pkl file")

    args = parser.parse_args()
    main(args)