"""
Stage 2 Preprocessing Pipeline for Support Integrity Auditor (SIA).

This script processes the pseudo-labeled CRM dataset generated in Stage 1 
and prepares it for fine-tuning a DeBERTa-v3-small model. It features text cleaning,
metadata label encoding, severity feature scaling, stratified splitting, and the
instantiation of a custom PyTorch Dataset class (`SIADataset`).

Artifacts are securely stored to prevent data leakage during train/eval workflows.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("stage2_preprocessing")


class SIADataset(Dataset):
    """
    Custom PyTorch Dataset class for the Support Integrity Auditor (SIA) Stage 2 model.
    Yields text sequences, categorical features, normalized structural metrics, and target labels.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the Dataset object.

        Args:
            df (pd.DataFrame): Dataframe containing fully processed text, encoded 
                               metadata, and normalized severity features.
        """
        self.texts = df["combined_text"].astype(str).tolist()
        self.channels = df["encoded_channel"].astype(int).tolist()
        self.domain_tiers = df["encoded_domain_tier"].astype(int).tolist()
        
        # Severity metrics arrays
        self.llm_severity = df["LLM_Severity"].astype(float).tolist()
        self.resolution_severity = df["Resolution_Severity"].astype(float).tolist()
        self.cluster_severity = df["Cluster_Severity"].astype(float).tolist()
        self.fused_severity = df["Fused_Severity"].astype(float).tolist()
        
        # Target labels
        self.labels = df["Mismatch_Label"].astype(int).tolist()

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the structured items corresponding to an index.

        Returns:
            Dict[str, Any]: Formatted data dictionary optimized for model fine-tuning.
        """
        return {
            "text": self.texts[idx],
            "channel": torch.tensor(self.channels[idx], dtype=torch.long),
            "domain_tier": torch.tensor(self.domain_tiers[idx], dtype=torch.long),
            "llm_severity": torch.tensor(self.llm_severity[idx], dtype=torch.float32),
            "resolution_severity": torch.tensor(self.resolution_severity[idx], dtype=torch.float32),
            "cluster_severity": torch.tensor(self.cluster_severity[idx], dtype=torch.float32),
            "fused_severity": torch.tensor(self.fused_severity[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Loads the pseudo-labeled CRM dataset and standardizes variant column structures.

    Args:
        filepath (str | Path): Path to the pseudo_labeled_tickets.csv file.

    Returns:
        pd.DataFrame: Loaded dataset with standardized column names.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Target data file not found at path: {path}")
        raise FileNotFoundError(f"Target data file not found at path: {path}")

    logger.info(f"Loading pseudo-labeled dataset from: {path}")
    df = pd.read_csv(path)
    
    # Standardize spaces to underscores for robust attribute lookups
    df.columns = [col.replace(" ", "_") for col in df.columns]
    logger.info(f"Successfully loaded {len(df)} entries. Standardized columns: {list(df.columns)}")
    return df


def clean_text(text: Any) -> str:
    """
    Cleans incoming textual data by lowercasing, stripping special characters, 
    and collapsing repeating space fragments.

    Args:
        text (Any): Input object raw sequence text.

    Returns:
        str: Fully normalized sequence string.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text_str = str(text).lower()
    # Strip everything except alphanumeric tokens and standard spaces
    text_str = re.sub(r"[^a-z0-9\s]", "", text_str)
    # Collapse multiple spaces or tabs into a singular space character
    normalized = re.sub(r"\s+", " ", text_str).strip()
    return normalized


def combine_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standardization and concats Subject and Description lines using a DeBERTa [SEP] token.

    Args:
        df (pd.DataFrame): Dataframe containing Subject and Description columns.

    Returns:
        pd.DataFrame: Dataframe containing the enriched 'combined_text' column.
    """
    df = df.copy()
    logger.info("Normalizing and combining Ticket Subject and Ticket Description fields...")
    
    clean_sub = df["Ticket_Subject"].fillna("").apply(clean_text)
    clean_desc = df["Ticket_Description"].fillna("").apply(clean_text)
    
    df["combined_text"] = clean_sub + " [SEP] " + clean_desc
    return df


def encode_metadata(
    df: pd.DataFrame, 
    channel_encoder: LabelEncoder | None = None, 
    domain_encoder: LabelEncoder | None = None
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Transforms string categorical columns into integer label values.
    Supports feeding pre-fit encoders to safely bypass test/validation data leakage.

    Args:
        df (pd.DataFrame): Target split DataFrame.
        channel_encoder (LabelEncoder, optional): Fitted encoder instance for channels.
        domain_encoder (LabelEncoder, optional): Fitted encoder instance for customer domains.

    Returns:
        Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]: Transformed DF and the tracking instances.
    """
    df = df.copy()
    
    # Defensive fallbacks for data field variants
    channel_col = "Ticket_Channel" if "Ticket_Channel" in df.columns else "Channel"
    domain_col = "Domain_Tier" if "Domain_Tier" in df.columns else "Product_Purchased"
    
    df[channel_col] = df[channel_col].fillna("Unknown").astype(str)
    df[domain_col] = df[domain_col].fillna("Unknown").astype(str)

    if channel_encoder is None:
        logger.info("Fitting new LabelEncoder instance for Ticket Channel features.")
        channel_encoder = LabelEncoder()
        df["encoded_channel"] = channel_encoder.fit_transform(df[channel_col])
    else:
        # Handle novel unseen tags securely under evaluation modes
        df["encoded_channel"] = df[channel_col].apply(
            lambda x: channel_encoder.transform([x])[0] if x in channel_encoder.classes_ else -1
        )

    if domain_encoder is None:
        logger.info("Fitting new LabelEncoder instance for Domain Tier features.")
        domain_encoder = LabelEncoder()
        df["encoded_domain_tier"] = domain_encoder.fit_transform(df[domain_col])
    else:
        df["encoded_domain_tier"] = df[domain_col].apply(
            lambda x: domain_encoder.transform([x])[0] if x in domain_encoder.classes_ else -1
        )

    return df, channel_encoder, domain_encoder


def scale_severity_features(
    df: pd.DataFrame, 
    scaler: StandardScaler | None = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalizes numerical severity signals using a standard feature scaling strategy.

    Args:
        df (pd.DataFrame): Target split DataFrame.
        scaler (StandardScaler, optional): Pre-fitted scaler instance for eval sets.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Transformed DataFrame and the tracking scaler instance.
    """
    df = df.copy()
    severity_cols = ["LLM_Severity", "Resolution_Severity", "Cluster_Severity", "Fused_Severity"]
    
    # Fill arbitrary sparse NaN values defensively if present
    for col in severity_cols:
        df[col] = df[col].fillna(0.5).astype(float)

    if scaler is None:
        logger.info("Fitting new StandardScaler instance across severity parameter metrics.")
        scaler = StandardScaler()
        df[severity_cols] = scaler.fit_transform(df[severity_cols])
    else:
        df[severity_cols] = scaler.transform(df[severity_cols])

    return df, scaler


def split_dataset(
    df: pd.DataFrame, 
    train_size: float = 0.70, 
    val_size: float = 0.15, 
    test_size: float = 0.15, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes a multi-stage stratified split maintaining target class balance.

    Args:
        df (pd.DataFrame): Complete base dataframe.
        train_size (float): Proportion target for training. Defaults to 0.70.
        val_size (float): Proportion target for validation. Defaults to 0.15.
        test_size (float): Proportion target for testing. Defaults to 0.15.
        random_state (int): Anchor seed reproducibility flag. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    logger.info("Initializing multi-stage stratified data splits...")
    
    # Guard calculation scaling logic
    assert np.isclose(train_size + val_size + test_size, 1.0), "Splits metrics must equal exactly 1.0"
    
    stratify_col = df["Mismatch_Label"]

    # Stage A: Separate test tracking profile block out from the primary cohort
    remainder_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Stage B: Separate training and validation files out from the balance remainder pool
    adjusted_val_size = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        remainder_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=remainder_df["Mismatch_Label"]
    )

    logger.info(f"Data separation successful. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df.copy(), val_df.copy(), test_df.copy()


def save_preprocessing_artifacts(
    channel_encoder: LabelEncoder, 
    domain_encoder: LabelEncoder, 
    severity_scaler: StandardScaler, 
    output_dir: str | Path = "models/preprocessing/"
) -> None:
    """
    Serializes tracking weights and parameters to disk for serving workflows.

    Args:
        channel_encoder (LabelEncoder): Fitted Channel mapping module.
        domain_encoder (LabelEncoder): Fitted Domain Tier mapping module.
        severity_scaler (StandardScaler): Fitted continuous feature scaler.
        output_dir (str | Path): Target storage directory location.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "channel_encoder.pkl": channel_encoder,
        "domain_encoder.pkl": domain_encoder,
        "severity_scaler.pkl": severity_scaler
    }

    for name, obj in artifacts.items():
        artifact_path = out_path / name
        with open(artifact_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved serialization checkpoint artifact: {artifact_path}")


def create_dataset(df: pd.DataFrame) -> SIADataset:
    """
    Helper instantiation constructor wrapping standard evaluation arrays inside PyTorch wrapper.

    Args:
        df (pd.DataFrame): Preprocessed subset Dataframe block.

    Returns:
        SIADataset: Initialized PyTorch dataset.
    """
    return SIADataset(df)


def prepare_stage2_data(filepath: str | Path) -> Tuple[SIADataset, SIADataset, SIADataset, Dict[str, Any]]:
    """
    Orchestrator pipeline executing linear tracking runs without causing data leakage.
    Fits state transformations *exclusively* on the train split, transforming validation/test afterwards.

    Args:
        filepath (str | Path): Input location pointing to 'pseudo_labeled_tickets.csv'.

    Returns:
        Tuple[SIADataset, SIADataset, SIADataset, Dict[str, Any]]: 
            Returns (train_dataset, val_dataset, test_dataset, metadata_artifacts_dict)
    """
    logger.info("Starting Stage 2 Production Preprocessing Pipeline Pipeline Workflow Execution.")
    
    # 1. Load configuration and text fields mapping structures
    raw_df = load_data(filepath)
    enriched_df = combine_text_fields(raw_df)
    
    # 2. Isolate tracks instantly via stratified splits to enforce boundaries
    train_df, val_df, test_df = split_dataset(enriched_df)
    
    # 3. Fit and transform training components to create primary references
    train_df, channel_enc, domain_enc = encode_metadata(train_df)
    train_df, sev_scaler = scale_severity_features(train_df)
    
    # 4. Transform validation metrics using training references (Leakage prevention)
    val_df, _, _ = encode_metadata(val_df, channel_encoder=channel_enc, domain_encoder=domain_enc)
    val_df, _ = scale_severity_features(val_df, scaler=sev_scaler)
    
    # 5. Transform test metrics using training references (Leakage prevention)
    test_df, _, _ = encode_metadata(test_df, channel_encoder=channel_enc, domain_encoder=domain_enc)
    test_df, _ = scale_severity_features(test_df, scaler=sev_scaler)
    
    # 6. Save tracking artifacts checkpoints to storage directories
    save_preprocessing_artifacts(channel_enc, domain_enc, sev_scaler)
    
    # 7. Map structures to torch objects
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df)
    
    metadata = {
        "channel_encoder": channel_enc,
        "domain_encoder": domain_enc,
        "severity_scaler": sev_scaler
    }
    
    logger.info("Stage 2 Preprocessing completely accomplished. Datasets prepared for DeBERTa tokenization steps.")
    return train_dataset, val_dataset, test_dataset, metadata


if __name__ == "__main__":
    # Internal validation test run matching the defaults tree structural configurations
    sample_path = Path("data/processed/pseudo_labeled_tickets.csv")
    if sample_path.exists():
        train_ds, val_ds, test_ds, meta = prepare_stage2_data(sample_path)
        logger.info(f"Execution test confirmation. First sequence item shape configuration payload: {train_ds[0]}")
    else:
        logger.warning(f"Standalone pipeline test run bypassed. Target file not located at '{sample_path}'.")