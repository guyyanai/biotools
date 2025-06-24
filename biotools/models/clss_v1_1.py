from typing import List, Tuple
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from transformers import EsmModel, EsmTokenizer


class CLSSv1_1(pl.LightningModule):
    def __init__(
        self,
        esm2_checkpoint: str,
        hidden_dim: int,
        learning_rate: float = 1e-3,
        init_temperature=np.log(1 / 0.07),
        should_learn_temperature: bool = True,
        random_sequence_stretches: bool = False,
        random_stretch_min_size: int = 30,
        should_load_esm3: bool = True,
    ):
        super(CLSSv1_1, self).__init__()
        self.save_hyperparameters()

        self.sequence_encoder, self.sequence_tokenizer = self.load_esm2(esm2_checkpoint)

        if should_load_esm3:
            self.structure_encoder = self.load_esm3()

        # Add a linear layer for projection
        self.sequence_projection_head = nn.Sequential(
            nn.ReLU(), nn.Linear(self.sequence_encoder.config.hidden_size, hidden_dim)
        )

        self.structure_projection_head = nn.Sequential(nn.Linear(1536, hidden_dim))

        self.learning_rate = learning_rate
        self.random_sequence_stretches = random_sequence_stretches
        self.random_stretch_min_size = random_stretch_min_size
        self.should_load_esm3 = should_load_esm3

        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )

        if not should_learn_temperature:
            self.temperature.requires_grad = False

    def load_esm2(self, checkpoint: str) -> Tuple[EsmModel, EsmTokenizer]:
        # Load the pre-trained ESM2 tokenizer & model
        tokenizer = EsmTokenizer.from_pretrained(checkpoint)
        model: EsmModel = EsmModel.from_pretrained(checkpoint)  # type: ignore

        # Disable unneeded parameters
        for parameter_name, parameter in list(model.named_parameters()):
            if "lm_head" in parameter_name:
                parameter.requires_grad = False

            if "contact_head" in parameter_name:
                parameter.requires_grad = False

        return model, tokenizer

    def load_esm3(self, checkpoint=ESM3_OPEN_SMALL) -> ESM3:
        model = ESM3.from_pretrained(checkpoint)

        for parameter in model.parameters():
            parameter.requires_grad = False

        return model

    def embed_sequences(self, sequences: List[str]) -> torch.Tensor:
        embedding_list = []

        for sequence in sequences:
            tokenized_sequence = self.sequence_tokenizer(
                sequence, return_tensors="pt"
            ).to(self.device)
            output = self.sequence_encoder(**tokenized_sequence)
            embedding = output.last_hidden_state.mean(dim=1)[0]
            embedding_list.append(embedding)

        esm_embeddings = torch.stack(embedding_list)
        embeddings = self.sequence_projection_head(esm_embeddings)
        normalized_embeddings = F.normalize(embeddings, dim=1)

        return normalized_embeddings

    def embed_structures(self, structures: List[torch.Tensor]) -> torch.Tensor:
        if self.should_load_esm3:
            raise Exception(
                "should_load_esm3 flag is off, please turn it on to embed structures using CLSS"
            )

        esm_proteins = [ESMProtein(coordinates=structure) for structure in structures]
        embedding_list = []

        for esm_protein in esm_proteins:
            protein_tensor = self.structure_encoder.encode(esm_protein)

            output = self.structure_encoder.forward_and_sample(
                protein_tensor, SamplingConfig(return_mean_embedding=True)
            )

            embedding_list.append(output.mean_embedding)

        esm_embeddings = torch.stack(embedding_list)
        embeddings = self.structure_projection_head(esm_embeddings)
        normalized_embeddings = F.normalize(embeddings, dim=1)

        return normalized_embeddings

    def sample_random_stretch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        sequence_length = int(attention_mask.count_nonzero().item())

        if sequence_length < self.random_stretch_min_size:
            start_index = 0
            substring_length = sequence_length
        else:
            start_index = int(
                torch.randint(
                    0, sequence_length - self.random_stretch_min_size + 1, (1,)
                ).item()
            )

            max_length = sequence_length - start_index
            substring_length = int(
                torch.randint(self.random_stretch_min_size, max_length + 1, (1,)).item()
            )

        stretch_input_ids = input_ids[start_index : start_index + substring_length]
        stretch_attention_mask = attention_mask[
            start_index : start_index + substring_length
        ]

        return stretch_input_ids, stretch_attention_mask, substring_length

    def forward(
        self,
        batched_input_ids: torch.Tensor,
        batched_attention_mask: torch.Tensor,
        batched_structure_embeddings: torch.Tensor,
    ):
        # Get the outputs from the ESM model
        sequence_outputs = []

        stretch_lengths = torch.zeros(batched_input_ids.shape[0])
        sequence_lengths = torch.zeros(batched_input_ids.shape[0])

        for index in range(batched_input_ids.shape[0]):
            input_ids = batched_input_ids[index]
            attention_mask = batched_attention_mask[index]

            sequence_length = attention_mask.count_nonzero().item()
            sequence_lengths[index] = sequence_length

            if self.random_sequence_stretches:
                input_ids, attention_mask, stretch_length = self.sample_random_stretch(
                    input_ids, attention_mask
                )
                stretch_lengths[index] = stretch_length

            sequence_output = self.sequence_encoder(
                input_ids=input_ids[:sequence_length].unsqueeze(0),
                attention_mask=attention_mask[:sequence_length].unsqueeze(0),
            )

            # Take the mean pooling of the last hidden state
            sequence_output = sequence_output.last_hidden_state.mean(dim=1)[0]

            sequence_outputs.append(sequence_output)

        if self.random_sequence_stretches:
            self.log("sequence_mean_length", sequence_lengths.mean(), logger=True)
            self.log("random_stretch_mean_length", stretch_lengths.mean(), logger=True)

        sequence_outputs = torch.stack(sequence_outputs)

        # Apply the projection head
        sequence_projections = self.sequence_projection_head(sequence_outputs)
        structure_projections = self.structure_projection_head(
            batched_structure_embeddings
        )

        return sequence_projections, structure_projections

    def gather_projections(self, projections: torch.Tensor) -> torch.Tensor:
        gathered = self.all_gather(projections, sync_grads=True)

        # Reshape tensors if necessary
        if isinstance(gathered, list):
            gathered = torch.cat(gathered, dim=0)
        else:
            gathered = gathered.view(-1, projections.size(-1))  # type: ignore

        return gathered

    def contrastive_loss(self, projections1: torch.Tensor, projections2: torch.Tensor):
        # Gather embeddings from all GPUs
        gathered_emb1 = self.gather_projections(projections1)
        gathered_emb2 = self.gather_projections(projections2)

        self.log("loss_total_samples", gathered_emb1.shape[0], prog_bar=True)

        # Normalize the projections
        projections1 = F.normalize(gathered_emb1, dim=1)
        projections2 = F.normalize(gathered_emb2, dim=1)

        # Compute cosine similarity
        similarities = torch.mm(projections1, projections2.T) / self.temperature.exp()

        # Labels for contrastive learning: diagonal elements should match
        labels = torch.arange(projections1.size(0), device=self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(similarities, labels)

        # Log individual components (example: log mean similarity of positive pairs)
        pos_similarity = similarities.detach().diag().mean()
        self.log("pos_similarity", pos_similarity.cpu(), prog_bar=True, logger=True)
        self.log(
            "temperature", self.temperature.detach().cpu(), prog_bar=True, logger=True
        )

        return loss

    def training_step(self, batch, batch_idx):
        # Assume batch contains paired sequences for contrastive learning
        input_ids, attention_mask, structure_embeddings = batch

        # Log the number of samples in this GPU's batch
        self.log(
            f"train_batch_size_per_gpu_{self.trainer.global_rank}",
            structure_embeddings.shape[0],
        )

        # Forward pass for both pairs
        sequence_projections, structure_projections = self(
            input_ids, attention_mask, structure_embeddings
        )

        # Compute contrastive loss
        loss = self.contrastive_loss(sequence_projections, structure_projections)
        self.log("train_loss", loss.detach(), prog_bar=True, logger=True)

        # Log additional metrics
        learning_rate = self.optimizers().param_groups[0]["lr"]  # type: ignore
        self.log("learning_rate", learning_rate, prog_bar=True)

        gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log(
            "gradient_norm", gradient_norm.detach().cpu(), prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # Assume batch contains paired sequences for contrastive learning
        input_ids, attention_mask, structure_embeddings = batch

        # Log the number of samples in this GPU's batch
        self.log(
            f"validation_batch_size_per_gpu_{self.trainer.global_rank}",
            structure_embeddings.shape[0],
        )

        # Forward pass for sequence and structure
        sequence_projections, structure_projections = self(
            input_ids, attention_mask, structure_embeddings
        )

        # Compute contrastive loss
        val_loss = self.contrastive_loss(sequence_projections, structure_projections)
        self.log(
            "val_loss", val_loss.detach(), prog_bar=True, logger=True, sync_dist=True
        )

        return val_loss

    def configure_optimizers(self):
        # Set up the optimizer
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # type: ignore

    def on_save_checkpoint(self, checkpoint):
        # Strip frozen model's weights before saving
        for key in list(checkpoint["state_dict"]):
            if key.startswith("structure_encoder"):
                del checkpoint["state_dict"][key]
