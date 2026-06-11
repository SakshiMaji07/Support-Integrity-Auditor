"""
Stage 2 Training Pipeline for the Support Integrity Auditor (SIA) Project.

Trains a fine-tuned, multimodal microsoft/deberta-v3-small classifier by combining 
contextual text embeddings with categorical and numerical ticket metadata.
Optimized for fast GPU execution using Automatic Mixed Precision (AMP) and 
dynamic batch padding.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

# Ensure script can locate stage2_preprocessing.py in the project structure
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from stage2_preprocessing import prepare_stage2_data, SIADataset
except ImportError:
    from src.stage2_preprocessing import prepare_stage2_data, SIADataset

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("stage2_training")


class SIAMultimodalModel(nn.Module):
    """
    Multimodal classification network fusing text sequence representations from 
    an unfrozen DeBERTa-v3 backbone with dense metadata features.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-small", num_labels: int = 2) -> None:
        super().__init__()
        logger.info(f"Loading DeBERTa-v3 backbone: {model_name}")
        self.deberta = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.deberta.config.hidden_size  # 768 dimensions for small
        # Text embedding size (768) + 6 explicit metadata and severity dimensions
        input_dim = hidden_size + 6 
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        channel: torch.Tensor,
        domain_tier: torch.Tensor,
        llm_severity: torch.Tensor,
        resolution_severity: torch.Tensor,
        cluster_severity: torch.Tensor,
        fused_severity: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass combining pooled textual representations and dense arrays."""
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pull first token [CLS] hidden state for pooled text sequence representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Concatenate metadata parameters horizontally [Batch Size, 6]
        extra_features = torch.stack([
            channel.float(),
            domain_tier.float(),
            llm_severity,
            resolution_severity,
            cluster_severity,
            fused_severity
        ], dim=1)
        
        combined_features = torch.cat((pooled_output, extra_features), dim=1)
        return self.classifier(combined_features)


class SIADynamicDataCollator:
    """
    Data Collator that performs dynamic batch padding on text fields 
    on-the-fly, reducing GPU sequence padding overhead.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        
        # Pad dynamically to the maximum sequence length inside this specific batch
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "channel": torch.stack([item["channel"] for item in batch]),
            "domain_tier": torch.stack([item["domain_tier"] for item in batch]),
            "llm_severity": torch.stack([item["llm_severity"] for item in batch]),
            "resolution_severity": torch.stack([item["resolution_severity"] for item in batch]),
            "cluster_severity": torch.stack([item["cluster_severity"] for item in batch]),
            "fused_severity": torch.stack([item["fused_severity"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch])
        }


def set_seed(seed: int = 42) -> None:
    """Sets reproducibility seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Reproducibility seed locked at: {seed}")


def create_dataset(filepath: str | Path) -> Tuple[SIADataset, SIADataset, SIADataset]:
    """
    Imports and instantiates structural dataframes via stage2_preprocessing pipeline.

    Args:
        filepath (str | Path): Path to the pseudo-labeled tickets.

    Returns:
        Tuple[SIADataset, SIADataset, SIADataset]: Train, validation, and test dataset instances.
    """
    logger.info("Extracting data partitions from stage2_preprocessing pipeline...")
    train_ds, val_ds, test_ds, _ = prepare_stage2_data(filepath)
    return train_ds, val_ds, test_ds


def compute_class_weights(train_dataset: SIADataset) -> torch.Tensor:
    """
    Calculates class balancing weights dynamically from the target labels array.

    Args:
        train_dataset (SIADataset): The active training subset.

    Returns:
        torch.Tensor: Weights tensor mapping the class inverse distribution frequencies.
    """
    labels = np.array(train_dataset.labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts)
    logger.info(f"Class counts: {class_counts}. Dynamic loss balancing weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_dataset: SIADataset,
    val_dataset: SIADataset,
    tokenizer: PreTrainedTokenizerBase,
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    patience: int = 2
) -> nn.Module:
    """
    Orchestrates the model training routine. Utilizes Automatic Mixed Precision (AMP)
    to optimize GPU execution speeds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Targeting training execution platform: {device}")
    model.to(device)

    collator = SIADynamicDataCollator(tokenizer=tokenizer)
    
    # Fast performance configuration: pin_memory=True speeds up transfer to CUDA GPUs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True)

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # AMP components initialization
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights_path = Path("models/preprocessing/temp_best_weights.pt")
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Map batch tokens to target runtime execution device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            channel = batch["channel"].to(device, non_blocking=True)
            domain_tier = batch["domain_tier"].to(device, non_blocking=True)
            llm_sev = batch["llm_severity"].to(device, non_blocking=True)
            res_sev = batch["resolution_severity"].to(device, non_blocking=True)
            clu_sev = batch["cluster_severity"].to(device, non_blocking=True)
            fus_sev = batch["fused_severity"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Cast forward pass into autocast mixed precision block for maximum GPU performance
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(
                    input_ids, attention_mask, channel, domain_tier,
                    llm_sev, res_sev, clu_sev, fus_sev
                )
                loss = criterion(logits, labels)

            # Backward pass using AMP scaled gradients
            scaler.scale(loss).backward()
            
            # Clip unscaled gradients to guard against gradient explosions
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(
                        batch["input_ids"].to(device, non_blocking=True),
                        batch["attention_mask"].to(device, non_blocking=True),
                        batch["channel"].to(device, non_blocking=True),
                        batch["domain_tier"].to(device, non_blocking=True),
                        batch["llm_severity"].to(device, non_blocking=True),
                        batch["resolution_severity"].to(device, non_blocking=True),
                        batch["cluster_severity"].to(device, non_blocking=True),
                        batch["fused_severity"].to(device, non_blocking=True)
                    )
                    v_loss = criterion(logits, batch["label"].to(device, non_blocking=True))
                total_val_loss += v_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Epoch {epoch}/{epochs} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Convergence evaluation & Early stopping execution tracks
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.warning(f"Early stopping condition triggered at epoch {epoch}.")
                break

    # Re-apply optimal historical parameters checkpoint if created
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        os.remove(best_weights_path)
        
    return model


def evaluate_model(model: nn.Module, test_dataset: SIADataset, tokenizer: PreTrainedTokenizerBase) -> Dict[str, float]:
    """
    Evaluates the model against the test dataset partition, logging key tracking metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    collator = SIADynamicDataCollator(tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collator, pin_memory=True)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                    batch["channel"].to(device, non_blocking=True),
                    batch["domain_tier"].to(device, non_blocking=True),
                    batch["llm_severity"].to(device, non_blocking=True),
                    batch["resolution_severity"].to(device, non_blocking=True),
                    batch["cluster_severity"].to(device, non_blocking=True),
                    batch["fused_severity"].to(device, non_blocking=True)
                )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

    logger.info(f"Final Test Evaluation Performance Metrics Profile: {metrics}")
    
    # Persist metrics profile to disk
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics


def save_model(model: nn.Module, tokenizer: PreTrainedTokenizerBase, output_dir: str | Path = "models/deberta_sia/") -> None:
    """Saves tokenizer patterns and structural state dictionary paths to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), out_path / "model.pt")
    model.deberta.config.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    logger.info(f"Model weights, configuration, and tokenizer successfully serialized to: {out_path}")


if __name__ == "__main__":
    DATA_PATH = Path("data/processed/pseudo_labeled_tickets.csv")
    MODEL_DIR = Path("models/deberta_sia/")
    
    set_seed(42)

    if not DATA_PATH.exists():
        logger.error(f"SIA Execution halted. Pseudo-labeled data source mapping file not found at: {DATA_PATH}")
        sys.exit(1)

    # 1. Pipeline Dataset Load Initialization
    train_data, val_data, test_data = create_dataset(DATA_PATH)
    
    # 2. Tokenizer Instance Mounting
    logger.info("Downloading/instantiating DeBERTa tokenizer parameters...")
    tokenizer_obj = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

    # 3. Model Topology Build Phase
    sia_network = SIAMultimodalModel("microsoft/deberta-v3-small")

    # 4. Multimodal Model Optimization Training Run
    trained_sia_network = train_model(
        model=sia_network,
        train_dataset=train_data,
        val_dataset=val_data,
        tokenizer=tokenizer_obj,
        epochs=5,
        batch_size=32,
        learning_rate=2e-5
    )

    # 5. Evaluate Holdout Validation Metrics Profile
    evaluate_model(trained_sia_network, test_data, tokenizer=tokenizer_obj)

    # 6. Save State Weights Checkpoints
    save_model(trained_sia_network, tokenizer_obj, MODEL_DIR)
    logger.info("Stage 2 Multimodal Model training execution pipeline completed successfully.")