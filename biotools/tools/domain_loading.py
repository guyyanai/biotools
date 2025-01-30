import os
import wget
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from typing import List

def read_aa_sequence_from_pdb(file_path: str):
    """
    Reads an amino acid sequence from a PDB file using a simple approach.
    
    Args:
        file_path (str): Path to the PDB file.
    
    Returns:
        str: The one-letter amino acid sequence extracted from the PDB file.
    """
    # Create a dictionary to map 3-letter amino acid codes to 1-letter codes
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    sequence = []  # List to store the sequence of amino acids
    seen_residues = set()  # Track already seen residues to avoid duplicates

    try:
        with open(file_path, 'r') as pdb_file:
            for line in pdb_file:
                if line.startswith('ATOM') and ' CA ' in line:  # Look for alpha carbon (CA) atoms
                    res_name = line[17:20].strip()  # Residue name (columns 18-20)
                    res_seq = line[22:26].strip()  # Residue sequence number (columns 23-26)
                    chain_id = line[21]  # Chain identifier (column 22)
                    unique_res_id = (chain_id, res_seq)  # Unique identifier for each residue

                    if res_name in three_to_one and unique_res_id not in seen_residues:
                        sequence.append(three_to_one[res_name])
                        seen_residues.add(unique_res_id)

    except Exception as e:
        print(f"An error occurred while reading the PDB file: {e}")

    return ''.join(sequence)


def download_pdb_file(domain_id: str, domain_type: str, local_file_path: str):
    if domain_type == 'CATH':
        pdb_url = f'http://www.cathdb.info/version/v4_3_0/api/rest/id/{domain_id}.pdb'
    elif domain_type == 'ECOD-ID':
        pdb_url = f'http://prodata.swmed.edu/ecod/af2_pdb/structure?id={domain_id}'
    elif domain_type == 'ECOD-UID':
        pdb_url = f'http://prodata.swmed.edu/ecod/af2_pdb/structure?uid={domain_id}'
    else:
        raise Exception(f'Failed to create remote pdb url!\nDomain ID: {domain_id}\nDomain Type: {domain_type}\n')
    
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    # Fetch the PDB file
    try:
        wget.download(pdb_url, local_file_path)
    except Exception as e:
        raise Exception(f'Failed to download pdb file!\nDomain ID: {domain_id}\nDomain Type: {domain_type}\nRemote URL: {pdb_url}\nException: ', e)
    
    return local_file_path

def load_domain_from_pdb(domain_id: str, structures_base_dir: str, domain_type: str, should_download_pdbs: bool, file_name_suffix = '.pdb'):
    
    if domain_type == 'ECOD-UID':
        pdb_file_path = os.path.join(structures_base_dir, domain_id[2:7], domain_id, f"{domain_id}{file_name_suffix}")
    else:
        pdb_file_path = os.path.join(structures_base_dir, f"{domain_id}{file_name_suffix}")
    
    if not os.path.exists(pdb_file_path):
        if not should_download_pdbs:
            raise Exception(f'Failed to find local pdb file!\nDomain ID: {domain_id}\nDomain Type: {domain_type}\nLocal URL: {pdb_file_path}')
        
        download_pdb_file(domain_id, domain_type, pdb_file_path)
    
    try:
        domain_chain = ProteinChain.from_pdb(pdb_file_path)
    except Exception as e:
        print(f'Failed to read pdb using ProteinChain, error: ', e)
        sequence = read_aa_sequence_from_pdb(pdb_file_path)
        return sequence, None
    
    load_domain = ESMProtein.from_protein_chain(domain_chain)

    sequence = load_domain.sequence
    coordinates = load_domain.coordinates
    
    return sequence, coordinates

def load_domains_from_pdb(domain_ids: List[str], structures_base_dir: str, domain_type: str, should_download_pdbs: bool, throw_if_failed: bool = False):
    sequences = []
    structures = []

    for domain_id in domain_ids:
        sequence, structure = load_domain_from_pdb(domain_id, structures_base_dir, domain_type, should_download_pdbs)

        if throw_if_failed:
            if sequence is None:
                raise Exception(f'Failed to load sequence of domain: {domain_id}')
            elif structure is None:
                raise Exception(f'Failed to load sequence of domain: {domain_id}')
        
        sequences.append(sequence)
        structures.append(structure)
    
    return sequences, structures