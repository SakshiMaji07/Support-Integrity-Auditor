"""
Restored implementation layer for stage2_preprocessing.py.
Handles decoupled raw features alignment joining textual fields and pseudo-annotations.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger("stage2_preprocessing")

class SIADataset(Dataset):
    """Encapsulates features for the Multimodal DeBERTa Classifier."""
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self.labels = self.df["Mismatch_Label"].astype(int).tolist() if "Mismatch_Label" in self.df.columns else [0] * len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        return {
            "combined_text": str(row.get("combined_text", "")),
            "channel": torch.tensor(row.get("Channel_Encoded", 0.0), dtype=torch.float32),
            "domain_tier": torch.tensor(row.get("Domain_Tier_Encoded", 0.0), dtype=torch.float32),
            "llm_severity": torch.tensor(row.get("LLM_Severity", 0.0), dtype=torch.float32),
            "resolution_severity": torch.tensor(row.get("Resolution_Severity", 0.0), dtype=torch.float32),
            "cluster_severity": torch.tensor(row.get("Cluster_Severity", 0.0), dtype=torch.float32),
            "fused_severity": torch.tensor(row.get("Inferred_Severity", 0.0), dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def prepare_stage2_data(
    filepath: str | Path,
    pseudo_labels_filepath: str | Path | None = None
) -> Tuple[SIADataset, SIADataset, SIADataset, Dict[str, Any]]:
    """Loads text attributes from primary path and marries them with secondary label metrics."""
    logger.info(f"Loading primary textual files from: {filepath}")
    df_primary = pd.read_csv(filepath)
    
    # Generate unified contextual processing fields if missing
    if "combined_text" not in df_primary.columns:
        subject = df_primary.get("Ticket_Subject", "")
        description = df_primary.get("Ticket_Description", "")
        df_primary["combined_text"] = "Subject: " + subject.astype(str) + " | Description: " + description.astype(str)

    if pseudo_labels_filepath and Path(pseudo_labels_filepath).exists():
        logger.info(f"Merging annotations from secondary source: {pseudo_labels_filepath}")
        df_labels = pd.read_csv(pseudo_labels_filepath)
        
        # Merge tracking components safely using Unique Ticket ID fields
        join_col = "Ticket_ID" if "Ticket_ID" in df_primary.columns and "Ticket_ID" in df_labels.columns else None
        if join_col:
            cols_to_use = list(df_labels.columns.difference(df_primary.columns)) + [join_col]
            df_primary = pd.merge(df_primary, df_labels[cols_to_use], on=join_col, how="inner")
        else:
            # Fallback to horizontal alignment if tracking indexes match perfectly
            for col in df_labels.columns:
                if col not in df_primary.columns:
                    df_primary[col] = df_labels[col]

    # Quick fill placeholders for mock tracking arrays
    for col, enc in [("Channel_Encoded", 0.0), ("Domain_Tier_Encoded", 0.0), 
                     ("LLM_Severity", 0.5), ("Resolution_Severity", 0.5), 
                     ("Cluster_Severity", 0.5), ("Inferred_Severity", 0.5),
                     ("Mismatch_Label", 0)]:
        if col not in df_primary.columns:
            df_primary[col] = enc

    # Train/Validation/Test Matrix Segmentation split allocations
    train_df, test_df = train_test_split(df_primary, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    return SIADataset(train_df), SIADataset(val_df), SIADataset(test_df), {}
