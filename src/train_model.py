"""
Stage 2 Training Pipeline for the Support Integrity Auditor (SIA) Project.
Trains a fine-tuned, multimodal microsoft/deberta-v3-small classifier.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

# Add parent directory to path for relative imports if executed standalone
_parent_dir = str(Path(__file__).resolve().parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from stage2_preprocessing import SIADataset, prepare_stage2_data

logger = logging.getLogger("stage2_training")


class SIAPreTokenizedDataset(SIADataset):
    """Pre-tokenized dataset for faster training loops."""

    def __init__(
        self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int = 512
    ) -> None:
        super().__init__(df)
        texts = df["combined_text"].astype(str).tolist()
        logger.info(f"Pre-tokenizing {len(texts)} text samples...")
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]
        logger.info(f"Pre-tokenization complete. Token shape: {self.input_ids.shape}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        item["input_ids"] = self.input_ids[idx]
        item["attention_mask"] = self.attention_mask[idx]
        return item


class SIAOptimizedCollator:
    """Optimized batch collator for training."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase | None = None) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "channel": torch.stack([item["channel"] for item in batch]),
            "domain_tier": torch.stack([item["domain_tier"] for item in batch]),
            "llm_severity": torch.stack([item["llm_severity"] for item in batch]),
            "resolution_severity": torch.stack(
                [item["resolution_severity"] for item in batch]
            ),
            "cluster_severity": torch.stack(
                [item["cluster_severity"] for item in batch]
            ),
            "fused_severity": torch.stack([item["fused_severity"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
        }


class SIAMultimodalModel(nn.Module):
    """Multimodal classification network fusing text with structural metadata."""

    def __init__(
        self, model_name: str = "microsoft/deberta-v3-small", num_labels: int = 2
    ) -> None:
        super().__init__()
        logger.info(f"Loading DeBERTa-v3 backbone: {model_name}")
        self.deberta = AutoModel.from_pretrained(model_name)

        hidden_size = self.deberta.config.hidden_size  # 768 for small variant
        input_dim = hidden_size + 6  # text pooled state + 6 metadata metrics

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )
        logger.info(
            f"SIAMultimodalModel initialized with input_dim={input_dim}, output_dim={num_labels}"
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
        fused_severity: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass combining pooled sequence vectors with numerical arrays."""
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        extra_features = torch.stack(
            [
                channel.float(),
                domain_tier.float(),
                llm_severity.float(),
                resolution_severity.float(),
                cluster_severity.float(),
                fused_severity.float(),
            ],
            dim=1,
        )

        combined_features = torch.cat((pooled_output, extra_features), dim=1)
        return self.classifier(combined_features)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Reproducibility seed locked at: {seed}")


def create_dataset(
    filepath: str | Path,
    pseudo_labels_filepath: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> Tuple[SIADataset, SIADataset, SIADataset, Dict[str, Any]]:
    """Imports and splits dataset states via underlying preprocessing interfaces."""
    logger.info(f"Extracting data partitions from stage2_preprocessing pipeline...")
    train_ds, val_ds, test_ds, artifacts = prepare_stage2_data(
        filepath, pseudo_labels_filepath, artifact_dir
    )
    return train_ds, val_ds, test_ds, artifacts


def compute_class_weights(train_dataset: SIADataset) -> torch.Tensor:
    """Calculates class balancing weights dynamically from target labels array."""
    labels = np.array(train_dataset.labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_dataset: SIADataset,
    val_dataset: SIADataset,
    tokenizer: PreTrainedTokenizerBase,
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    patience: int = 2,
    accumulation_steps: int = 4,
    num_workers: int = 0,
) -> nn.Module:
    """Optimized multi-signal cross-entropy classification training orchestration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Targeting training execution platform: {device}")
    model.to(device)

    # Pre-tokenize elements to optimize training throughput
    train_pretok = (
        SIAPreTokenizedDataset(train_dataset.df, tokenizer)
        if hasattr(train_dataset, "df")
        else train_dataset
    )
    val_pretok = (
        SIAPreTokenizedDataset(val_dataset.df, tokenizer)
        if hasattr(val_dataset, "df")
        else val_dataset
    )

    collator = SIAOptimizedCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_pretok,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_pretok,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        num_workers=num_workers,
    )

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    total_steps = (len(train_loader) * epochs) // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights_path = Path("models/preprocessing/temp_best_weights.pt")
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            channel = batch["channel"].to(device, non_blocking=True)
            domain_tier = batch["domain_tier"].to(device, non_blocking=True)
            llm_sev = batch["llm_severity"].to(device, non_blocking=True)
            res_sev = batch["resolution_severity"].to(device, non_blocking=True)
            clu_sev = batch["cluster_severity"].to(device, non_blocking=True)
            fus_sev = batch["fused_severity"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(
                    input_ids,
                    attention_mask,
                    channel,
                    domain_tier,
                    llm_sev,
                    res_sev,
                    clu_sev,
                    fus_sev,
                )
                loss = criterion(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps

        avg_train_loss = total_train_loss / len(train_loader)

        # Periodic dataset evaluations tracking loop updates
        if epoch % 2 == 1 or epoch == epochs:
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
                            batch["fused_severity"].to(device, non_blocking=True),
                        )
                        v_loss = criterion(
                            logits, batch["label"].to(device, non_blocking=True)
                        )
                    total_val_loss += v_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            logger.info(
                f"Epoch {epoch}/{epochs} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_weights_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.warning(f"Early stopping condition triggered at epoch {epoch}.")
                    break

    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        os.remove(best_weights_path)
        logger.info("Restored best model weights configuration layer state.")

    return model


def train(
    train_data: SIADataset,
    val_data: SIADataset,
    test_data: SIADataset,
    artifacts: Dict[str, Any],
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Wrapper entry-point interface function handling model training loop cycles."""
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    model = SIAMultimodalModel("microsoft/deberta-v3-small", num_labels=2)

    trained_model = train_model(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        tokenizer=tokenizer,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=2,
    )

    metrics = evaluate_model(trained_model, test_data, tokenizer)
    save_model(trained_model, tokenizer, artifacts, "models/deberta_sia/")
    return trained_model, metrics


def evaluate_model(
    model: nn.Module, test_dataset: SIADataset, tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Any]:
    """Comprehensive evaluation with all required metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    collator = SIAOptimizedCollator(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )

    all_preds = []
    all_labels = []
    all_probs = []

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
                    batch["fused_severity"].to(device, non_blocking=True),
                )
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ===== COMPREHENSIVE METRICS =====
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = (
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Classification report
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=["Consistent", "Mismatch"],
        output_dict=True,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "precision_consistent": float(precision_per_class[0]),
        "precision_mismatch": float(precision_per_class[1]),
        "recall_consistent": float(recall_per_class[0]),
        "recall_mismatch": float(recall_per_class[1]),
        "f1_consistent": float(f1_per_class[0]),
        "f1_mismatch": float(f1_per_class[1]),
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
    }

    logger.info(f"Final Test Evaluation Performance Metrics Profile: {metrics}")

    # Display metrics
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION METRICS")
    print("=" * 70)
    print(f"\nBinary Classification Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"  Class 0 (Consistent):")
    print(f"    - Precision: {precision_per_class[0]:.4f}")
    print(f"    - Recall:    {recall_per_class[0]:.4f}")
    print(f"    - F1 Score:  {f1_per_class[0]:.4f}")
    print(f"  Class 1 (Mismatch):")
    print(f"    - Precision: {precision_per_class[1]:.4f}")
    print(f"    - Recall:    {recall_per_class[1]:.4f}")
    print(f"    - F1 Score:  {f1_per_class[1]:.4f}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")
    print(
        f"\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=['Consistent', 'Mismatch'])}"
    )
    print("=" * 70 + "\n")

    return metrics


def save_model(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    artifacts: Dict[str, Any],
    output_dir: str | Path = "models/deberta_sia/",
) -> None:
    """Saves tokenizer, model, and preprocessing artifacts to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), out_path / "model.pt")
    model.deberta.config.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    # Save preprocessing artifacts
    if artifacts:
        for artifact_name, artifact_obj in artifacts.items():
            artifact_path = out_path / f"{artifact_name}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact_obj, f)

    logger.info(f"Model parameters successfully serialized to: {out_path}")


if __name__ == "__main__":
    DATA_PATH = Path("data/processed/processed.csv")
    PSEUDO_LABELS_PATH = Path("data/processed/pseudo_labeled_tickets.csv")
    ARTIFACT_DIR = Path("models/preprocessing/")

    set_seed(42)
    train_data, val_data, test_data, artifacts_dict = create_dataset(
        DATA_PATH, PSEUDO_LABELS_PATH, ARTIFACT_DIR
    )
    tokenizer_obj = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    sia_network = SIAMultimodalModel("microsoft/deberta-v3-small")

    trained_sia_network = train_model(
        model=sia_network,
        train_dataset=train_data,
        val_dataset=val_data,
        tokenizer=tokenizer_obj,
        epochs=5,
        batch_size=32,
        learning_rate=2e-5,
    )
    evaluate_model(trained_sia_network, test_data, tokenizer=tokenizer_obj)
    save_model(trained_sia_network, tokenizer_obj, artifacts_dict, "models/deberta_sia/")
