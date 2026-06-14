from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict

import sys
from pathlib import Path

# Ensure src is in path
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    
# Import stage 1 processing and labeling routines without duplicating logic
from src.stage1_preprocessing import load_data, preprocess_dataframe, save_processed_data
from src.pseudo_labels import generate_pseudo_labels
from src.stage2_preprocessing import prepare_stage2_data
from src.train_model import train

logger = logging.getLogger(__name__)

def run_stage1(
    raw_data_path: str | Path | None = None,
    processed_data_path: str | Path | None = None,
    pseudo_labels_path: str | Path | None = None,
    summary_output_path: str | Path | None = None,
) -> None:
    """Orchestrates Stage 1 of the pipeline: preprocessing, pseudo-labeling, and metrics tracking."""
    logger.info("Starting Pipeline Stage 1: Preprocessing & Pseudo-Label Generation")

    # Set default paths if none provided
    project_root = Path(__file__).resolve().parent
    if processed_data_path is None:
        processed_data_path = project_root / "data" / "processed" / "processed.csv"
    if pseudo_labels_path is None:
        pseudo_labels_path = project_root / "data" / "processed" / "pseudo_labeled_tickets.csv"
    if summary_output_path is None:
        summary_output_path = project_root / "outputs" / "stage1_summary.json"

    processed_data_path = Path(processed_data_path)
    pseudo_labels_path = Path(pseudo_labels_path)
    summary_output_path = Path(summary_output_path)

    try:
        # 1. Preprocess raw data
        logger.info("Step 1/3: Preprocessing raw dataset...")
        df_raw = load_data(raw_data_path)
        df_processed = preprocess_dataframe(df_raw)
        save_processed_data(df_processed, processed_data_path)

        # 2. Generate Pseudo-labels
        logger.info("Step 2/3: Generating self-supervised pseudo-labels...")
        df_pseudo = generate_pseudo_labels(
            input_path=processed_data_path, 
            output_path=pseudo_labels_path
        )

        # 3. Compute Stage 1 Statistics & Metrics
        logger.info("Step 3/3: Evaluating signal agreement and generating audit summary...")
        ticket_count = len(df_pseudo)
        mismatch_rate = float(df_pseudo["Mismatch_Label"].mean())

        # Extract and format the mismatch distribution (counts of 0s and 1s)
        mismatch_counts = df_pseudo["Mismatch_Label"].value_counts().to_dict()
        mismatch_distribution = {str(k): int(v) for k, v in mismatch_counts.items()}

        # Extract and format the inferred severity distribution
        severity_counts = df_pseudo["Inferred_Severity"].value_counts().to_dict()
        inferred_severity_distribution = {str(k): int(v) for k, v in severity_counts.items()}

        # Map continuous signals to discrete severity buckets using the pipeline's cutoff thresholds
        def _score_to_bucket(series: pd.Series) -> np.ndarray:
            return np.where(series <= 0.3, 1,
                   np.where(series <= 0.55, 2,
                   np.where(series <= 0.75, 3, 4)))

        llm_buckets = _score_to_bucket(df_pseudo["LLM_Severity"])
        res_buckets = _score_to_bucket(df_pseudo["Resolution_Severity"])
        cluster_buckets = _score_to_bucket(df_pseudo["Cluster_Severity"])

        # Compute signal agreement metrics (matching buckets / total records)
        llm_vs_resolution = float(np.mean(llm_buckets == res_buckets))
        llm_vs_cluster = float(np.mean(llm_buckets == cluster_buckets))
        resolution_vs_cluster = float(np.mean(res_buckets == cluster_buckets))

        summary_data = {
            "ticket_count": ticket_count,
            "mismatch_rate": mismatch_rate,
            "mismatch_distribution": mismatch_distribution,
            "inferred_severity_distribution": inferred_severity_distribution,
            "signal_agreement": {
                "llm_vs_resolution": llm_vs_resolution,
                "llm_vs_cluster": llm_vs_cluster,
                "resolution_vs_cluster": resolution_vs_cluster,
            },
        }

        # Write execution summary out to file
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)

        logger.info("Stage 1 execution summary successfully generated at %s", summary_output_path)
        logger.info("Stage 1 execution completed successfully.")

    except Exception as error:
        logger.exception("Pipeline processing failure during Stage 1: %s", error)
        raise

def run_stage2(
    ticket_data_path: str | Path | None = None,
    pseudo_labels_path: str | Path | None = None,
    metrics_output_path: str | Path | None = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Coordinates data loading, multi-signal tabular pre-formatting, and model training steps.

    CRITICAL CHANGE: Now uses evidence.csv/processed.csv as primary ticket data source (with text),
    and optionally merges pseudo-labels for annotations.

    Args:
        ticket_data_path (str | Path | None): Path to evidence.csv or processed.csv (PRIMARY text source).
        pseudo_labels_path (str | Path | None): Path to pseudo_labeled_tickets.csv (SECONDARY for labels).
        metrics_output_path (str | Path | None): Local path target for JSON checkpointing.
        random_seed (int): Reproducibility state variable passed down backend modules.

    Returns:
        Dict[str, Any]: Matrix containing evaluation scores parsed out of the training suite.
    """
    logger.info("Initializing Support Integrity Auditor (SIA) Stage 2 Framework Execution...")
    logger.info("Data flow: PRIMARY=[evidence.csv/processed.csv with text] + SECONDARY=[pseudo_labels.csv for annotations]")

    # Establish localized target workspace layouts
    project_root = Path(__file__).resolve().parent
    if ticket_data_path is None:
        # PRIMARY data source: use evidence.csv or processed.csv
        ticket_data_path = project_root / "data" / "processed" / "evidence.csv"
    if pseudo_labels_path is None:
        # SECONDARY data source: pseudo-labels for annotations
        pseudo_labels_path = project_root / "data" / "processed" / "pseudo_labeled_tickets.csv"
    if metrics_output_path is None:
        metrics_output_path = project_root / "outputs" / "metrics.json"

    ticket_data_path = Path(ticket_data_path)
    pseudo_labels_path = Path(pseudo_labels_path)
    metrics_output_path = Path(metrics_output_path)

    # Step 1: Validate primary data source exists
    if not ticket_data_path.exists():
        logger.error(f"Execution halted. PRIMARY ticket data missing at: {ticket_data_path}")
        logger.error(f"Expected evidence.csv or processed.csv with Ticket_Subject and Ticket_Description columns.")
        raise FileNotFoundError(f"Required PRIMARY ticket data missing: {ticket_data_path}")

    logger.info(f"Step 1/4: Loading PRIMARY ticket data from: {ticket_data_path}")
    logger.info(f"          (Optional SECONDARY labels from: {pseudo_labels_path})")

    # Step 2: Offload text processing, mapping, and tokenization to stage2_preprocessing
    logger.info("Step 2/4: Transferring records data matrix to feature processing framework...")
    train_dataset, val_dataset, test_dataset, preprocessing_artifacts = prepare_stage2_data(
        filepath=ticket_data_path,
        pseudo_labels_filepath=pseudo_labels_path if pseudo_labels_path.exists() else None
    )

    # Step 3: Extract and print statistical insights about data splits and class profiles
    logger.info("Step 3/4: Compiling descriptive metrics on feature subsets...")
    
    # Check if datasets have mismatch labels for distribution analysis
    try:
        if len(train_dataset) > 0 and hasattr(train_dataset, 'df'):
            raw_df = train_dataset.df
            if "Mismatch_Label" in raw_df.columns:
                distribution = raw_df["Mismatch_Label"].value_counts().to_dict()
                dist_str = f"Consistent (0): {distribution.get(0, 0)} | Mismatch (1): {distribution.get(1, 0)}"
            else:
                dist_str = "Label distribution: Mismatch_Label column not found in training data."
        else:
            dist_str = "Label distribution: Unable to compute from datasets."
    except Exception as e:
        logger.warning(f"Could not compute label distribution: {e}")
        dist_str = "Label distribution: Unable to compute."

    print("\n" + "="*60)
    print("       SUPPORT INTEGRITY AUDITOR DATASET METRICS STATUS")
    print("="*60)
    print(f" Training Sample Subsets Block Matrix : {len(train_dataset):,}")
    print(f" Validation Optimization Line Space   : {len(val_dataset):,}")
    print(f" Test Evaluation Boundary Array Size : {len(test_dataset):,}")
    print(f" Global Baseline Class Distribution   : {dist_str}")
    print("="*60 + "\n")

    # Step 4: Transfer finalized datasets to train_model suite for model tuning
    logger.info("Step 4/4: Passing PyTorch dataset references to the DeBERTa training routine...")
    trained_model, performance_metrics = train(
        train_data=train_dataset,
        val_data=val_dataset,
        test_data=test_dataset,
        artifacts=preprocessing_artifacts,
        seed=random_seed
    )

    # Save metrics block to structured JSON format
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as metrics_file:
        json.dump(performance_metrics, metrics_file, indent=4, ensure_ascii=False)
    
    logger.info(f"Evaluation metrics snapshot successfully exported to file: {metrics_output_path}")

    # Display final execution summary in formatted view layout
    print("\n" + "="*60)
    print("          STAGE 2 DEBERTA MODEL VALIDATION METRICS")
    print("="*60)
    for key, score in performance_metrics.items():
        print(f" • {key.title().ljust(15)} : {float(score) * 100:.2f}%")
    print("="*60 + "\n")

    logger.info("SIA Orchestration Layer pipeline run executed cleanly without runtime exceptions.")
    return performance_metrics

def main() -> None:
    """Main entry point configuration parsing execution options."""
    parser = argparse.ArgumentParser(
        description="Orchestration suite runner for Support Integrity Auditor Project Workflow Pipelines."
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Optional path override pointing to custom location for raw data processing targeting.",
    )
    parser.add_argument(
        "--processed-data-path",
        type=str,
        default=None,
        help="Target filepath specification for saving out Intermediate Normalized DataFrame stages.",
    )
    parser.add_argument(
        "--pseudo-labels-path",
        type=str,
        default=None,
        help="Destination target override for writing pseudo-labeled records data tables.",
    )
    parser.add_argument(
        "--summary-output-path",
        type=str,
        default=None,
        help="Destination target output path for tracking quality assurance execution run profiles.",
    )
    parser.add_argument(
        "--ticket-data-path",
        type=str,
        default=None,
        help="[Stage 2] Path to PRIMARY ticket data source (evidence.csv or processed.csv with text).",
    )
    parser.add_argument(
        "--metrics-output-path",
        type=str,
        default=None,
        help="Output path for Stage 2 metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Main random seed value selection to guarantee reproducibility across underlying runtime engines.",
    )
    args = parser.parse_args()

    # Configure operational logger layout properties cleanly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set underlying numpy seeds to ensure deterministic execution steps inside modules
    np.random.seed(args.seed)

    run_stage1(
        raw_data_path=args.raw_data_path,
        processed_data_path=args.processed_data_path,
        pseudo_labels_path=args.pseudo_labels_path,
        summary_output_path=args.summary_output_path,
    )
    
    run_stage2(
        ticket_data_path=args.ticket_data_path,
        pseudo_labels_path=args.pseudo_labels_path,
        metrics_output_path=args.metrics_output_path,
        random_seed=args.seed
    )
