import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import EsmModel


class CLSSNonLin(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        learning_rate: float = 1e-3,
        init_temperature=torch.log(torch.tensor(1 / 0.07, dtype=torch.float32)),
        should_learn_temperature: bool = False,
        random_sequence_stretches: bool = False,
        random_stretch_min_size: int = 30,
    ):
        super(CLSSNonLin, self).__init__()
        self.save_hyperparameters()

        # Load the pre-trained ESM2 model
        self.esm2_model: EsmModel = EsmModel.from_pretrained(model_name)  # type: ignore

        for parameter_name, parameter in list(self.esm2_model.named_parameters()):
            if "lm_head" in parameter_name:
                parameter.requires_grad = False

            if "contact_head" in parameter_name:
                parameter.requires_grad = False

        # Add a linear layer for projection
        self.esm2_projection_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.esm2_model.config.hidden_size, hidden_dim),
        )

        self.esm3_projection_head = nn.Sequential(
            nn.ReLU(), nn.Linear(1536, hidden_dim)
        )

        self.temperature = nn.Parameter(init_temperature)

        if not should_learn_temperature:
            self.temperature.requires_grad = False

        # Learning rate
        self.learning_rate = learning_rate
        self.random_sequence_stretches = random_sequence_stretches
        self.random_stretch_min_size = random_stretch_min_size
        self.model_name = model_name

    def forward(
        self, batched_input_ids, batched_attention_mask, batched_structure_embeddings
    ):
        # Get the outputs from the ESM model
        sequence_outputs = []
        # structure_outputs = []

        stretch_lengths = torch.zeros(batched_input_ids.shape[0])
        sequence_lengths = torch.zeros(batched_input_ids.shape[0])

        for index in range(batched_input_ids.shape[0]):
            input_ids = batched_input_ids[index]
            attention_mask = batched_attention_mask[index]

            sequence_length = attention_mask.count_nonzero().item()
            sequence_lengths[index] = sequence_length

            if self.random_sequence_stretches:
                if sequence_length < self.random_stretch_min_size:
                    start_index = 0
                    substring_length = sequence_length
                else:
                    start_index = torch.randint(
                        0, sequence_length - self.random_stretch_min_size + 1, (1,)
                    ).item()
                    max_length = sequence_length - start_index
                    substring_length = torch.randint(
                        self.random_stretch_min_size, max_length + 1, (1,)
                    ).item()

                stretch_lengths[index] = substring_length
                input_ids = input_ids[start_index : start_index + substring_length]
                attention_mask = attention_mask[
                    start_index : start_index + substring_length
                ]

            sequence_output = self.esm2_model(
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
        # structure_outputs = torch.stack(structure_outputs)

        # Apply the projection head
        sequence_projections = self.esm2_projection_head(sequence_outputs)
        structure_projections = self.esm3_projection_head(batched_structure_embeddings)

        return sequence_projections, structure_projections

    def contrastive_loss(self, projections1, projections2, temperature: float = 0.5):
        # Gather embeddings from all GPUs
        gathered_emb1 = self.all_gather(projections1)
        gathered_emb2 = self.all_gather(projections2)

        # Reshape tensors if necessary
        if isinstance(gathered_emb1, list):
            gathered_emb1 = torch.cat(gathered_emb1, dim=0)
        else:
            gathered_emb1 = gathered_emb1.view(-1, projections1.size(-1))  # type: ignore

        if isinstance(gathered_emb2, list):
            gathered_emb2 = torch.cat(gathered_emb2, dim=0)
        else:
            gathered_emb2 = gathered_emb2.view(-1, projections2.size(-1))  # type: ignore

        self.log("loss_total_samples", gathered_emb1.shape[0], prog_bar=True)

        # Normalize the projections
        projections1 = F.normalize(projections1, dim=1)
        projections2 = F.normalize(projections2, dim=1)

        # Compute cosine similarity
        similarities = torch.mm(projections1, projections2.T)
        scaled_similarities = similarities * self.temperature.exp()

        # Labels for contrastive learning: diagonal elements should match
        labels = torch.arange(projections1.size(0), device=self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scaled_similarities, labels)

        # Log individual components (example: log mean similarity of positive pairs)
        pos_similarity = similarities.detach().diag().mean().cpu()
        scaled_pos_similarity = scaled_similarities.detach().diag().mean().cpu()

        self.log("pos_similarity", pos_similarity, prog_bar=True, logger=True)
        self.log("scaled_pos_similarity", scaled_pos_similarity, logger=True)

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
            prog_bar=True,
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
        self.log("learning_rate", learning_rate, prog_bar=True, logger=True)

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
            prog_bar=True,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # type: ignore

        return optimizer
