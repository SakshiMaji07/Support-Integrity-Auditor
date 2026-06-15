"""
Stage 2 Preprocessing Pipeline for Support Integrity Auditor.

Handles data loading, categorical encoding, feature scaling, dataset creation,
and train/val/test splitting with proper artifact preservation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger("stage2_preprocessing")


class SIADataset(Dataset):
    """Encapsulates features for the Multimodal DeBERTa Classifier."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self.labels = (
            self.df["Mismatch_Label"].astype(int).tolist()
            if "Mismatch_Label" in self.df.columns
            else [0] * len(self.df)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "input_ids": None,  # Will be filled by collator
            "attention_mask": None,  # Will be filled by collator
            "combined_text": str(row.get("combined_text", "")),
            "channel": torch.tensor(
                row.get("Channel_Encoded", 0), dtype=torch.long
            ),
            "domain_tier": torch.tensor(
                row.get("Domain_Tier_Encoded", 0), dtype=torch.long
            ),
            "llm_severity": torch.tensor(
                row.get("LLM_Severity", 0.5), dtype=torch.float32
            ),
            "resolution_severity": torch.tensor(
                row.get("Resolution_Severity", 0.5), dtype=torch.float32
            ),
            "cluster_severity": torch.tensor(
                row.get("Cluster_Severity", 0.5), dtype=torch.float32
            ),
            "fused_severity": torch.tensor(
                row.get("Inferred_Severity", 0.5), dtype=torch.float32
            ),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _create_encoders(
    df: pd.DataFrame, artifact_dir: Path | None = None
) -> Tuple[LabelEncoder, LabelEncoder, StandardScaler]:
    """Create and optionally save categorical encoders and severity scaler."""
    artifact_dir = Path(artifact_dir) if artifact_dir else Path("models/preprocessing/")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Channel encoder
    channel_col = next(
        (col for col in df.columns if "channel" in col.lower()), "Ticket_Channel"
    )
    channel_encoder = LabelEncoder()
    channels = df[channel_col].fillna("Unknown").astype(str).unique()
    channel_encoder.fit(channels)

    # Domain tier encoder
    domain_col = next(
        (col for col in df.columns if "domain" in col.lower() and "tier" in col.lower()),
        "Domain_Tier",
    )
    domain_encoder = LabelEncoder()
    domains = df[domain_col].fillna("Unknown").astype(str).unique()
    domain_encoder.fit(domains)

    # Severity scaler
    severity_cols = [
        "LLM_Severity",
        "Resolution_Severity",
        "Cluster_Severity",
        "Inferred_Severity",
    ]
    severity_data = df[severity_cols].fillna(0.5).values
    severity_scaler = StandardScaler()
    severity_scaler.fit(severity_data)

    # Save artifacts
    with open(artifact_dir / "channel_encoder.pkl", "wb") as f:
        pickle.dump(channel_encoder, f)
    with open(artifact_dir / "domain_encoder.pkl", "wb") as f:
        pickle.dump(domain_encoder, f)
    with open(artifact_dir / "severity_scaler.pkl", "wb") as f:
        pickle.dump(severity_scaler, f)

    logger.info(f"Artifacts saved to {artifact_dir}")
    return channel_encoder, domain_encoder, severity_scaler


def _encode_and_scale(
    df: pd.DataFrame,
    channel_encoder: LabelEncoder,
    domain_encoder: LabelEncoder,
    severity_scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply encoding and scaling to dataframe."""
    df = df.copy()

    # Encode categorical features
    channel_col = next(
        (col for col in df.columns if "channel" in col.lower()), "Ticket_Channel"
    )
    domain_col = next(
        (col for col in df.columns if "domain" in col.lower() and "tier" in col.lower()),
        "Domain_Tier",
    )

    df["Channel_Encoded"] = (
        df[channel_col]
        .fillna("Unknown")
        .astype(str)
        .apply(
            lambda x: channel_encoder.transform([x])[0]
            if x in channel_encoder.classes_
            else -1
        )
    )
    df["Domain_Tier_Encoded"] = (
        df[domain_col]
        .fillna("Unknown")
        .astype(str)
        .apply(
            lambda x: domain_encoder.transform([x])[0]
            if x in domain_encoder.classes_
            else -1
        )
    )

    # Scale severity features
    severity_cols = [
        "LLM_Severity",
        "Resolution_Severity",
        "Cluster_Severity",
        "Inferred_Severity",
    ]
    severity_data = df[severity_cols].fillna(0.5).values
    scaled_severities = severity_scaler.transform(severity_data)

    for i, col in enumerate(severity_cols):
        df[col] = scaled_severities[:, i]

    return df


def prepare_stage2_data(
    filepath: str | Path,
    pseudo_labels_filepath: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> Tuple[SIADataset, SIADataset, SIADataset, Dict[str, Any]]:
    """
    Loads text and pseudo-label data, encodes features, splits into train/val/test.

    Args:
        filepath: Path to processed CSV with text fields
        pseudo_labels_filepath: Path to pseudo-labeled output from Stage 1
        artifact_dir: Directory to save/load encoders and scalers

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, artifacts_dict)
    """
    artifact_dir = Path(artifact_dir) if artifact_dir else Path("models/preprocessing/")

    logger.info(f"Loading primary ticket data from: {filepath}")
    df_primary = pd.read_csv(filepath)

    # Generate combined text if missing
    if "combined_text" not in df_primary.columns:
        subject = df_primary.get("Ticket_Subject", "")
        description = df_primary.get("Ticket_Description", "")
        df_primary["combined_text"] = (
            "Subject: "
            + subject.astype(str)
            + " | Description: "
            + description.astype(str)
        )

    # Merge pseudo-labels if provided
    if pseudo_labels_filepath and Path(pseudo_labels_filepath).exists():
        logger.info(f"Merging pseudo-labels from: {pseudo_labels_filepath}")
        df_labels = pd.read_csv(pseudo_labels_filepath)

        # Merge on Ticket_ID
        join_col = None
        if "Ticket_ID" in df_primary.columns and "Ticket_ID" in df_labels.columns:
            join_col = "Ticket_ID"

        if join_col:
            cols_to_use = list(
                df_labels.columns.difference(df_primary.columns)
            ) + [join_col]
            df_primary = pd.merge(
                df_primary, df_labels[cols_to_use], on=join_col, how="left"
            )
        else:
            # Fallback: concatenate new columns
            for col in df_labels.columns:
                if col not in df_primary.columns:
                    df_primary[col] = df_labels[col]

    # Ensure required columns exist
    required_cols = [
        "Ticket_Channel",
        "Domain_Tier",
        "LLM_Severity",
        "Resolution_Severity",
        "Cluster_Severity",
        "Inferred_Severity",
    ]
    for col in required_cols:
        if col not in df_primary.columns:
            if "Severity" in col:
                df_primary[col] = 0.5
            else:
                df_primary[col] = "Unknown"

    # Ensure Mismatch_Label exists
    if "Mismatch_Label" not in df_primary.columns:
        df_primary["Mismatch_Label"] = 0

    # Create encoders and scalers
    channel_encoder, domain_encoder, severity_scaler = _create_encoders(
        df_primary, artifact_dir
    )

    # Apply encoding and scaling
    df_primary = _encode_and_scale(
        df_primary, channel_encoder, domain_encoder, severity_scaler
    )

    # Train/Val/Test split with stratification
    train_df, test_df = train_test_split(
        df_primary,
        test_size=0.2,
        random_state=42,
        stratify=df_primary["Mismatch_Label"],
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["Mismatch_Label"]
    )

    logger.info(
        f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    artifacts = {
        "channel_encoder": channel_encoder,
        "domain_encoder": domain_encoder,
        "severity_scaler": severity_scaler,
    }

    return (
        SIADataset(train_df),
        SIADataset(val_df),
        SIADataset(test_df),
        artifacts,
    )
