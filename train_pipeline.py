from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Import stage 1 processing and labeling routines without duplicating logic
from stage1_preprocessing import load_data, preprocess_dataframe, save_processed_data
from src.pseudo_labels import generate_pseudo_labels

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


if __name__ == "__main__":
    main()