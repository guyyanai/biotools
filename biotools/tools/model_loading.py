import os
import torch
from transformers import EsmTokenizer, EsmModel
from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL
from models.clss import CLSS
from models.ProTrek.model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

def load_weights_from_checkpoint(checkpoint_path: str, weights_key_prefix: str):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Search for the ESM2 model weights in the checkpoint
    weights = {}
    for key in checkpoint['state_dict']:
        if key.startswith(weights_key_prefix + '.'):  # Adjust this based on your model's naming convention
            weights['.'.join(key.split('.')[1:])] = checkpoint['state_dict'][key]

    if weights is {}:
        raise ValueError(f'Weights not found in the checkpoint!\nKey Prefix: {weights_key_prefix}')

    return weights

def load_esm2_from_checkpoint(esm2_config: str, checkpoint_path: str, weights_key_prefix: str):
    
    # Check if the checkpoint isn't local
    if checkpoint_path.startswith('facebook/esm2'):
        if esm2_config == checkpoint_path:
            return EsmModel.from_pretrained(esm2_config)
        else:
            raise Exception(f'Failed to load ESM2 from remote checkpoint!\nConfig checkpoint: {esm2_config}\nWeights checkpoint: {checkpoint_path}')
    
    # Initialize model using correct config
    esm2_model = EsmModel.from_pretrained(esm2_config)
    
    # Load ESM2 weights from checkpoint
    esm2_weights = load_weights_from_checkpoint(checkpoint_path, weights_key_prefix)
    
    # Load weights into ESM2 model
    esm2_model.load_state_dict(esm2_weights)
    
    return esm2_model

def load_clss(clss_checkpoint: str, device = None):
    if device is None:
        device = torch.device('cuda')
    
    is_pre_trained = clss_checkpoint.startswith("facebook/esm2")
    if is_pre_trained:
        print(f"Using pre-trained model: {clss_checkpoint}")
        contrastive_transformer = CLSS(clss_checkpoint, 1)
    else:
        print(f"Using trained model: {clss_checkpoint}")
        contrastive_transformer = CLSS.load_from_checkpoint(
            clss_checkpoint, strict=False
        )

    esm2 = contrastive_transformer.esm2_model.to(device)
    esm2_projection_head = (
        contrastive_transformer.esm2_projection_head.to(device)
        if not is_pre_trained
        else None
    )
    esm2_tokenizer = EsmTokenizer.from_pretrained(contrastive_transformer.model_name)

    esm3_projection_head = (
        contrastive_transformer.esm3_projection_head.to(device)
        if not is_pre_trained
        else None
    )

    return esm2, esm2_tokenizer, esm2_projection_head, esm3_projection_head

def load_esm3(checkpoint=ESM3_OPEN_SMALL):
    return ESM3.from_pretrained(checkpoint)


def load_protrek(size = "650M", protrek_weights_dir='/local/protrek/weights'):
    if size == "650M":
        config = {
            "protein_config": os.path.join(protrek_weights_dir, 'ProTrek_650M_UniRef50/esm2_t33_650M_UR50D'),
            "text_config": os.path.join(protrek_weights_dir, 'ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'),
            "structure_config": os.path.join(protrek_weights_dir, 'ProTrek_650M_UniRef50/foldseek_t30_150M'),
            "load_protein_pretrained": False,
            "load_text_pretrained": False,
            "from_checkpoint": os.path.join(protrek_weights_dir, 'ProTrek_650M_UniRef50/ProTrek_650M_UniRef50.pt'),
        }
    elif size == "35M":
        config = {
            "protein_config": os.path.join(protrek_weights_dir, 'ProTrek_35M_UniRef50/esm2_t12_35M_UR50D'),
            "text_config": os.path.join(protrek_weights_dir, 'ProTrek_35M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'),
            "structure_config": os.path.join(protrek_weights_dir, 'ProTrek_35M_UniRef50/foldseek_t12_35M'),
            "load_protein_pretrained": False,
            "load_text_pretrained": False,
            "from_checkpoint": os.path.join(protrek_weights_dir, 'ProTrek_35M_UniRef50/ProTrek_35M_UniRef50.pt'),
        }
    else:
        raise Exception(f'Failed to load ProTrek model, the currently supported sizes are: 35M, 650M. Received size: {size}')

    return ProTrekTrimodalModel(**config).to('cuda')
