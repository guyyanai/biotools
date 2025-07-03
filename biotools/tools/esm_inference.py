import torch
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from typing import List
from transformers import EsmTokenizer, EsmModel
from tqdm import tqdm


def esm3_infer_esm_protein(
    esm_proteins: List[ESMProtein],
    esm3: ESM3,
    projection_head: torch.nn.Sequential,
    should_normalize=False,
    move_to_cpu=False,
    use_tqdm=True,
    tqdm_desc="Inferring proteins using ESM3...",
):
    embeddings = []

    for esm_protein in tqdm(esm_proteins, desc=tqdm_desc) if use_tqdm else esm_proteins:
        protein_tensor = esm3.encode(esm_protein)
        with torch.no_grad():
            result = esm3.forward_and_sample(
                protein_tensor, SamplingConfig(return_mean_embedding=True)
            )

        embedding = result.mean_embedding
        embeddings.append(embedding)

    embeddings_tensor = torch.stack(embeddings)

    if projection_head:
        with torch.no_grad():
            embeddings_tensor = projection_head(embeddings_tensor)

    if should_normalize:
        embeddings_tensor = F.normalize(embeddings_tensor, dim=1)

    if move_to_cpu:
        cpu_embeddings = embeddings_tensor.cpu()
        del embeddings_tensor
        return cpu_embeddings

    return embeddings_tensor


def esm2_infer_sequences(
    sequences: List[str],
    esm2: EsmModel,
    tokenizer: EsmTokenizer,
    projection_head: torch.nn.Sequential,
    should_normalize=False,
    device=None,
    use_tqdm=True,
    move_to_cpu=False,
    tqdm_desc="Inferring sequences using ESM2...",
):
    if device is None:
        device = torch.device("cuda")

    embeddings = []

    for sequence in tqdm(sequences, desc=tqdm_desc) if use_tqdm else sequences:
        tokenized = tokenizer(sequence, return_tensors="pt").to(device)

        with torch.no_grad():
            result = esm2(**tokenized)

        embedding = result.last_hidden_state.mean(dim=1)[0]
        embeddings.append(embedding)

    embeddings_tensor = torch.stack(embeddings)

    if projection_head:
        with torch.no_grad():
            embeddings_tensor = projection_head(embeddings_tensor)

    if should_normalize:
        embeddings_tensor = F.normalize(embeddings_tensor, dim=1)

    if move_to_cpu:
        cpu_embeddings = embeddings_tensor.cpu()
        del embeddings_tensor
        return cpu_embeddings

    return embeddings_tensor


def esm3_infer_structures(
    structures: List[torch.Tensor],
    esm3: ESM3,
    projection_head: torch.nn.Sequential,
    should_normalize=False,
    move_to_cpu=False,
    use_tqdm=True,
    tqdm_desc="Inferring structures using ESM3...",
):
    esm_proteins = [ESMProtein(coordinates=structure) for structure in structures]
    return esm3_infer_esm_protein(
        esm_proteins,
        esm3,
        projection_head,
        should_normalize,
        move_to_cpu,
        use_tqdm,
        tqdm_desc,
    )


def esm3_infer_sequences(
    sequences: List[str],
    esm3: ESM3,
    projection_head: torch.nn.Sequential,
    should_normalize=False,
    move_to_cpu=False,
    use_tqdm=True,
    tqdm_desc="Inferring sequences using ESM3...",
):
    esm_proteins = [ESMProtein(sequence=sequence) for sequence in sequences]
    return esm3_infer_esm_protein(
        esm_proteins,
        esm3,
        projection_head,
        should_normalize,
        move_to_cpu,
        use_tqdm,
        tqdm_desc,
    )


def esm3_infer_both(
    sequences: List[str],
    structures: List[torch.Tensor],
    esm3: ESM3,
    projection_head: torch.nn.Sequential,
    should_normalize=False,
    move_to_cpu=False,
    use_tqdm=True,
    tqdm_desc="Inferring sequences+structures using ESM3...",
):
    esm_proteins = [
        ESMProtein(sequence=sequence, coordinates=structure)
        for sequence, structure in zip(sequences, structures)
    ]
    return esm3_infer_esm_protein(
        esm_proteins,
        esm3,
        projection_head,
        should_normalize,
        move_to_cpu,
        use_tqdm,
        tqdm_desc,
    )
