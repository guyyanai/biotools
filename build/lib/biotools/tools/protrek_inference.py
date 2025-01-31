import os
from typing import List
import torch
from models.ProTrek.model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from models.ProTrek.utils.foldseek_util import get_struc_seq

def infer_structures_protrek(
    structures_paths: List[str],
    protrek: ProTrekTrimodalModel,
    foldseek_path = "/local/protrek/foldseek",
):
    embeddings = []
    invalid_domains = []

    for structure_path in structures_paths:

        seqs = get_struc_seq(foldseek_path, structure_path)
        seqs = seqs[list(seqs.keys())[0]]

        foldseek_seq = seqs[1].lower()

        if foldseek_seq is None:
            invalid_domains.append(os.path.basename(structure_path))
            continue

        with torch.no_grad():
            result = protrek.get_structure_repr([foldseek_seq])

        embedding = result[0]
        embeddings.append(embedding)

    embeddings = torch.stack(embeddings)

    if len(invalid_domains) > 0:
        print('ProTrek structure invalid domains: ', invalid_domains)

    return embeddings

def infer_sequences_protrek(
    sequences: List[str],
    protrek: ProTrekTrimodalModel,
):
    embeddings = []

    for sequence in sequences:

        with torch.no_grad():
            result = protrek.get_protein_repr([sequence])

        embedding = result[0]
        embeddings.append(embedding)

    embeddings = torch.stack(embeddings)

    return embeddings