import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser 
import numpy as np


protein_list = [key.upper() for key in protein_letters_3to1.keys()]

def extract_pdb_data(pdb_file):       
    parser=PDBParser(QUIET=True)
    structure = parser.get_structure("protein_structure", pdb_file)
  
    data = {
        "atom_name": [],
        "resname": [],
        "residue": [],
        "type": [],
        "chain": [],
        "x": [],
        "y": [],
        "z": [],
        "element": []
    }

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in ["A", "C", "G", "U"]:
                    rna_atoms = [(atom.get_name(), residue.get_resname().strip(), residue.get_id()[1], "rna", chain.get_id(), atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2], atom.element) for atom in residue]
                    data["atom_name"].extend([atom[0] for atom in rna_atoms])
                    data["resname"].extend([atom[1] for atom in rna_atoms])
                    data["residue"].extend([atom[2] for atom in rna_atoms])
                    data["type"].extend([atom[3] for atom in rna_atoms])
                    data["chain"].extend([atom[4] for atom in rna_atoms])
                    data["x"].extend([atom[5] for atom in rna_atoms])
                    data["y"].extend([atom[6] for atom in rna_atoms])
                    data["z"].extend([atom[7] for atom in rna_atoms])
                    data["element"].extend([atom[8] for atom in rna_atoms])
                elif residue.get_resname().strip() in protein_list:
                    protein_atoms = [(atom.get_name(), residue.get_resname().strip(), residue.get_id()[1], "protein", chain.get_id(), atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2], atom.element) for atom in residue]
                    data["atom_name"].extend([atom[0] for atom in protein_atoms])
                    data["resname"].extend([atom[1] for atom in protein_atoms])
                    data["residue"].extend([atom[2] for atom in protein_atoms])
                    data["type"].extend([atom[3] for atom in protein_atoms])
                    data["chain"].extend([atom[4] for atom in protein_atoms])
                    data["x"].extend([atom[5] for atom in protein_atoms])
                    data["y"].extend([atom[6] for atom in protein_atoms])
                    data["z"].extend([atom[7] for atom in protein_atoms])
                    data["element"].extend([atom[8] for atom in protein_atoms])
    
    # Check if the data contains both RNA and protein; if not, return
    if 'rna' not in data['type'] or 'protein' not in data['type']: 
        # print("RNA or protein not found")   
        return None
    
    data=pd.DataFrame(data)    


    return data 



def df_to_pdb(df, output_file, model_num=1):
    """
    Convert DataFrame to PDB format and write to file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing atomic coordinates with columns:
        atom_name, resname, residue, type, chain, x, y, z
    output_file : str
        Path to output PDB file
    model_num : int
        Model number to use in PDB file
    """
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"MODEL     {model_num}\n")
        
        # Write atomic coordinates
        atom_num = 1
        for idx, row in df.iterrows():
            # Format atom name with proper spacing
            atom_name = row['atom_name'].ljust(4)
            if len(atom_name) < 4:
                atom_name = ' ' + atom_name
                
            # Format residue name with proper spacing
            resname = row['resname'].rjust(3)
            
            # Create PDB line in standard format
            # ATOM/HETATM num atomname resname chain resnum x y z occupancy temp_factor element
            line = (f"{'ATOM':6s}"  # Record type
                   f"{atom_num:5d}"  # Atom serial number
                   f" {atom_name:<4s}"  # Atom name
                   f" {resname:3s}"  # Residue name
                   f" {row['chain']:1s}"  # Chain identifier
                   f"{int(row['residue']):4d}"  # Residue sequence number
                   f"    "  # Code for insertion of residues
                   f"{row['x']:8.3f}"  # X coordinate
                   f"{row['y']:8.3f}"  # Y coordinate
                   f"{row['z']:8.3f}"  # Z coordinate
                   f"  1.00"  # Occupancy
                   f"  0.00"  # Temperature factor
                   f"          "  # Blank space
                   f"{row['atom_name'][0]:>2s}"  # Element symbol
                   f"\n")
            
            f.write(line)
            atom_num += 1
        
        # Write footer
        f.write("ENDMDL\n")
        f.write("END\n")



 




def  align_dataframes(df1, df2, match_score=1, mismatch_penalty=-1, gap_penalty=-2, debug=False):
    """
    Implement Needleman-Wunsch algorithm with robust error handling.
    
    Parameters:
    df1, df2 (pd.DataFrame): DataFrames containing sequence information with 'resname' column
    match_score (int): Score for matching residues
    mismatch_penalty (int): Penalty for mismatched residues
    gap_penalty (int): Penalty for introducing a gap
    debug (bool): Whether to print debug information
    
    Returns:
    tuple: (aligned_df1, aligned_df2) - DataFrames with optimal global alignment
    """
    try:
        # Get sequences
        seq1 = df1['resname'].tolist()
        seq2 = df2['resname'].tolist()
        
        if debug:
            print(f"Length of sequence 1: {len(seq1)}")
            print(f"Length of sequence 2: {len(seq2)}")
            print("Sequence 1:", seq1)
            print("Sequence 2:", seq2)
        
        # Initialize the scoring matrix
        n, m = len(seq1), len(seq2)
        score_matrix = np.zeros((n + 1, m + 1))
        traceback = np.zeros((n + 1, m + 1), dtype=int)
        
        # Initialize first row and column
        score_matrix[:, 0] = np.arange(n + 1) * gap_penalty
        score_matrix[0, :] = np.arange(m + 1) * gap_penalty
        
        # Fill the scoring matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = score_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
                delete = score_matrix[i-1, j] + gap_penalty
                insert = score_matrix[i, j-1] + gap_penalty
                
                score_matrix[i, j] = max(match, delete, insert)
                
                if score_matrix[i, j] == match:
                    traceback[i, j] = 0
                elif score_matrix[i, j] == delete:
                    traceback[i, j] = 1
                else:
                    traceback[i, j] = 2
        
        # Traceback
        aligned_indices1 = []
        aligned_indices2 = []
        i, j = n, m
        
        while i > 0 and j > 0:  # Changed condition to prevent out of bounds
            current_trace = traceback[i, j]
            
            if current_trace == 0:  # Match/mismatch
                aligned_indices1.insert(0, i-1)
                aligned_indices2.insert(0, j-1)
                i -= 1
                j -= 1
            elif current_trace == 1:  # Gap in seq2
                i -= 1
            else:  # Gap in seq1
                j -= 1
        
        # Create aligned DataFrames
        aligned_df1 = df1.iloc[aligned_indices1].reset_index(drop=True)
        aligned_df2 = df2.iloc[aligned_indices2].reset_index(drop=True)
        
        if debug:
            print(f"Length of aligned sequence 1: {len(aligned_df1)}")
            print(f"Length of aligned sequence 2: {len(aligned_df2)}")
            print("Aligned sequence 1:", aligned_df1['resname'].tolist())
            print("Aligned sequence 2:", aligned_df2['resname'].tolist())
        
        # Verify alignment
        assert len(aligned_df1) == len(aligned_df2), "Alignment lengths do not match"
        assert all(aligned_df1['resname'] == aligned_df2['resname']), "Sequences not properly aligned"

        return aligned_df1, aligned_df2
    
    except Exception as e:
        print(f"Error during alignment: {str(e)}")
        # Return a fallback alignment using common subsequence
        return fallback_alignment(df1, df2)

def fallback_alignment(df1, df2):
    """
    Fallback alignment method when Needleman-Wunsch fails.
    Uses longest common subsequence approach.
    """
    seq1 = df1['resname'].tolist()
    seq2 = df2['resname'].tolist()
    
    # Find common elements while preserving order
    aligned_indices1 = []
    aligned_indices2 = []
    
    j = 0
    for i, res1 in enumerate(seq1):
        while j < len(seq2):
            if res1 == seq2[j]:
                aligned_indices1.append(i)
                aligned_indices2.append(j)
                j += 1
                break
            j += 1
    
    return (df1.iloc[aligned_indices1].reset_index(drop=True),
            df2.iloc[aligned_indices2].reset_index(drop=True))