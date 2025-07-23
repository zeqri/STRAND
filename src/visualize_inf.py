from fpdf import FPDF
import pandas as pd

import os
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime


from evaluation.compute_rmsd import   RMSDComputer
from helpers.helpers import extract_pdb_data ,align_dataframes 


rmsd_computer = RMSDComputer() 

def create_rmsd_report(complex_rmsd_values, min_rmsd_values, plot_label, output_path="rmsd_report.pdf"):
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
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
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
        int_pdb_files = [p for p in af3_dir_path.iterdir() if pattern.match(p.name) and not p.name.endswith("-0.pdb")]

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
        
        # if filename.upper() == "8EDJ":    
        #     print(f"{plot_label} - min diffused crmsd 8EDJ: {min_rmsd}") 
        #     print(f"{plot_label} - AF3 pred crmsd 8EDJ: {complex_rmsd}") 
        # if filename.upper() == "7VKL": 
        #     print(f"{plot_label} - min diffused crmsd 7VKL: {min_rmsd}") 
        #     print(f"{plot_label} - AF3 pred crmsd 7VKL: {complex_rmsd}") 
        
        min_and_complex_rmsd.append((min_rmsd, complex_rmsd))
        labels.append(filename)
        min_rmsd_filenames.append(min_rmsd_filename)

    # Compute overall percentage change
    total_complex_rmsd = sum(complex_rmsd for _, complex_rmsd in min_and_complex_rmsd)
    total_min_rmsd = sum(min_rmsd for min_rmsd, _ in min_and_complex_rmsd)

    if total_complex_rmsd != 0:
        overall_percentage_change = ((total_complex_rmsd - total_min_rmsd) / total_complex_rmsd) * 100
    else:
        overall_percentage_change = 0

    
    print(f"{plot_label} - Overall Percentage Change: {overall_percentage_change:.2f}%")
    print(f"{plot_label} - min rmsd filenames: {min_rmsd_filenames}")
    print(f"{plot_label} - complex rmsd values: {complex_rmsd_values}")
    print(f"{plot_label} - min rmsd values: {min_rmsd_values}")
    print(f"{plot_label} - labels: {labels}")
    
    create_rmsd_report(complex_rmsd_values, min_rmsd_values, "not_Xray", pdf_rep_path)


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


if __name__ == "__main__":
    

    gt_path = "/Users/muhsenalzzaqry/Desktop/DiffDock-PR/datasets/RNAPRO/rna_pro_Gt_residue"
    pdf_path="results/plots" 
    report_path="results/reports"     
    dir_path="/Users/muhsenalzzaqry/Desktop/DiffDock-PR/visualization_af3/tr_0.1/AF3_XRAY_RANDOMIZATION_TR_MAX_0.1_230/epoch-0"

    os.makedirs(pdf_path, exist_ok=True)
    os.makedirs(report_path, exist_ok=True)  
    path_configs = [

    {

        'dir_path': dir_path,   
        'plot_label': 'STRAND-tr+rot',
        'output_filename':  f'{pdf_path}/TR_ROT_NOT_XRAY.pdf', #TR-ROT ,x
        'report_path':  f'{report_path}/TR_ROT_NOT_XRAY.pdf'
    }

]

    for config in path_configs:
        print(f"\nProcessing {config['plot_label']}...")
        process_and_plot_path(
        config['dir_path'],
        config['plot_label'],
        config['output_filename'],
        gt_path,
        config['report_path']   
       )
