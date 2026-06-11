"""
Dossier Generator Pipeline for Support Integrity Auditor (SIA).

Extracts and correlates stage 1 signals, ticket text, metadata vectors, and model 
predictions to build structured, zero-hallucination compliance audits for mismatched tickets.
Determines whether a violation qualifies as a 'Hidden Crisis' or a 'False Alarm'.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Setup structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dossier_generator")

# Priority to Severity Numeric Mapping Matrix
PRIORITY_TO_SEVERITY_MAP = {
    "P1": 4, "P2": 3, "P3": 2, "P4": 1,
    "CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1
}


def determine_mismatch_type(assigned_priority: Any, inferred_severity: Any) -> Tuple[str, int]:
    """
    Evaluates the priority-severity spread vector to establish operational category types.

    Args:
        assigned_priority (Any): String or numeric tracking representation of the ticket priority.
        inferred_severity (Any): Numeric value representing the audited operational severity (1-4).

    Returns:
        Tuple[str, int]: (mismatch_type string, numeric severity_delta gap value)
    """
    # Normalize inputs to continuous numeric spaces safely
    p_str = str(assigned_priority).strip().upper()
    p_num = PRIORITY_TO_SEVERITY_MAP.get(p_str, None)
    
    if p_num is None:
        try:
            p_num = int(float(assigned_priority))
        except (ValueError, TypeError):
            p_num = 2  # Default to baseline mapping if unparseable
            
    try:
        s_num = int(float(inferred_severity))
    except (ValueError, TypeError):
        s_num = 2

    # Vector Delta calculation (Inferred Severity minus Assigned Priority representation)
    delta = s_num - p_num
    
    if delta > 0:
        # audited severity is higher than operational assigned lane -> Hidden SLA breach risk
        mismatch_type = "Hidden Crisis"
    elif delta < 0:
        # audited severity is lower than lane -> Resource allocation inefficiency
        mismatch_type = "False Alarm"
    else:
        mismatch_type = "Consistent"
        
    return mismatch_type, delta


def generate_feature_evidence(ticket_row: pd.Series) -> List[str]:
    """
    Extracts explicit data fragments from verified features without interpolation or extrapolation.

    Args:
        ticket_row (pd.Series): Single unified dataframe line item containing target ticket dimensions.

    Returns:
        List[str]: Array of grounded, factual evidence observations.
    """
    evidence: List[str] = []
    
    # 1. Ticket Text & Extracted Structural Keyword Anchors
    text_subject = str(ticket_row.get("Ticket_Subject", ticket_row.get("Ticket Subject", ""))).strip()
    if text_subject:
        evidence.append(f"Ticket Subject text states: '{text_subject}'")
        
    # 2. Metadata Channels and Domain Layer Parameters
    channel = str(ticket_row.get("Ticket_Channel", ticket_row.get("Ticket Channel", "Unknown"))).strip()
    domain = str(ticket_row.get("Domain_Tier", ticket_row.get("Domain Tier", "Unknown"))).strip()
    evidence.append(f"Ingress Channel identified as '{channel}' via Client Domain Tier '{domain}'.")

    # 3. Stage 1 Severity Signals Breakdown
    signals = {
        "LLM Severity": ticket_row.get("llm_severity", ticket_row.get("LLM_Severity", None)),
        "Resolution Severity": ticket_row.get("resolution_severity", ticket_row.get("Resolution_Severity", None)),
        "Cluster Severity": ticket_row.get("cluster_severity", ticket_row.get("Cluster_Severity", None)),
        "Fused Severity": ticket_row.get("fused_severity", ticket_row.get("Fused_Severity", None))
    }
    
    signal_strings = []
    for sig_name, sig_val in signals.items():
        if pd.notna(sig_val):
            signal_strings.append(f"{sig_name}: {float(sig_val):.2f}")
            
    if signal_strings:
        evidence.append(f"Stage 1 quantitative signals calculated as: {', '.join(signal_strings)}.")

    return evidence


def generate_constraint_analysis(ticket_row: pd.Series, mismatch_type: str, delta: int) -> str:
    """
    Formulates a strict, non-hallucinated analytical summary explaining the SLA conflict context.

    Args:
        ticket_row (pd.Series): Underlying ticket record series.
        mismatch_type (str): Categorized state string ('Hidden Crisis' or 'False Alarm').
        delta (int): Numeric separation metric showing deviation gaps.

    Returns:
        str: Grounded textual assertion outlining specific organizational impact variables.
    """
    p_lvl = str(ticket_row.get("Priority_Level", ticket_row.get("Priority Level", ticket_row.get("assigned_priority", "Unknown")))).strip()
    inf_sev = str(ticket_row.get("Inferred_Severity", ticket_row.get("Inferred Severity", "Unknown"))).strip()
    domain = str(ticket_row.get("Domain_Tier", ticket_row.get("Domain Tier", "Standard"))).strip()

    if mismatch_type == "Hidden Crisis":
        return (
            f"SLA Audit Conflict: Assigned priority is set at low-tier baseline '{p_lvl}', "
            f"but multi-signal model synthesis computes an actual operational severity index of '{inf_sev}' (Delta: +{delta}). "
            f"This operational delta presents an unmitigated SLA breach exposure for the customer account '{domain}'."
        )
    elif mismatch_type == "False Alarm":
        return (
            f"Resource Allocation Conflict: Ticket is routed inside high-priority lane '{p_lvl}', "
            f"but structural attributes evaluate to an audited severity level of '{inf_sev}' (Delta: {delta}). "
            f"This over-allocation of resources degrades engineering capacity and standard triage operations."
        )
    return "Audit indicates structural parity exists between assigned tracking lane configurations and model inference markers."


def build_dossier(ticket_row: pd.Series) -> Dict[str, Any]:
    """
    Assembles a structurally valid audit dossier schema for downstream consuming web components.

    Args:
        ticket_row (pd.Series): Merged system dataset data point.

    Returns:
        Dict[str, Any]: Grounded dictionary payload structure mapping the target compliance schema.
    """
    t_id = str(ticket_row.get("Ticket_ID", ticket_row.get("Ticket ID", "UNKNOWN_ID")))
    p_lvl = str(ticket_row.get("Priority_Level", ticket_row.get("Priority Level", "Unknown"))).strip()
    inf_sev = str(ticket_row.get("Inferred_Severity", ticket_row.get("Inferred Severity", "Unknown"))).strip()
    
    # Extract inference engine confidence if compiled during prediction phases
    confidence = ticket_row.get("Confidence", ticket_row.get("confidence", "1.00"))
    try:
        conf_str = f"{float(confidence):.4f}"
    except (ValueError, TypeError):
        conf_str = "1.0000"

    # Evaluate categorizations and text structures
    mismatch_type, delta = determine_mismatch_type(p_lvl, inf_sev)
    feature_evidence = generate_feature_evidence(ticket_row)
    constraint_analysis = generate_constraint_analysis(ticket_row, mismatch_type, delta)

    # Factory assembly exactly following specifications
    return {
        "ticket_id": t_id,
        "assigned_priority": p_lvl,
        "inferred_severity": inf_sev,
        "mismatch_type": mismatch_type,
        "severity_delta": str(delta),
        "feature_evidence": feature_evidence,
        "constraint_analysis": constraint_analysis,
        "confidence": conf_str
    }


def generate_dossiers_pipeline(
    predictions_path: Path | str = "outputs/predictions.csv",
    pseudo_labels_path: Path | str = "data/processed/pseudo_labeled_tickets.csv",
    output_json_path: Path | str = "outputs/mismatch_dossiers.json"
) -> List[Dict[str, Any]]:
    """
    Executes structural merges on upstream data frames to find mismatches and compile audited dossiers.
    """
    pred_file = Path(predictions_path)
    label_file = Path(pseudo_labels_path)
    out_file = Path(output_json_path)

    if not pred_file.exists() or not label_file.exists():
        logger.error(f"Dossier generator halted. Missing required files: {pred_file} or {label_file}")
        return []

    logger.info("Ingesting inference predictions and structural stage 1 pseudo labels...")
    df_preds = pd.read_csv(pred_file)
    df_labels = pd.read_csv(label_file)

    # Synchronize keys before performing merge sequence
    if "Ticket_ID" in df_preds.columns and "Ticket_ID" in df_labels.columns:
        merge_key = "Ticket_ID"
    elif "Ticket ID" in df_preds.columns and "Ticket ID" in df_labels.columns:
        merge_key = "Ticket ID"
    else:
        # Fallback indexing sync if column titles diverge completely across runtime scripts
        df_preds["merge_idx"] = df_preds.index
        df_labels["merge_idx"] = df_labels.index
        merge_key = "merge_idx"

    # Drop colliding columns from predictions frame to keep original source variables uncorrupted
    cols_to_drop = [col for col in df_preds.columns if col in df_labels.columns and col != merge_key]
    df_preds_clean = df_preds.drop(columns=cols_to_drop)

    merged_df = pd.merge(df_labels, df_preds_clean, on=merge_key, how="inner")
    logger.info(f"Successfully integrated {len(merged_df)} ticket telemetry matrix profiles.")

    # Filter for targets identified as priority mismatches
    # Fallback to structural calculation if explicit 'Prediction' column is missing or unpopulated
    if "Prediction" in merged_df.columns:
        mismatch_subset = merged_df[merged_df["Prediction"] == 1].copy()
    else:
        logger.warning("'Prediction' column unavailable. Computing categories using rule-based fallback vector steps instead.")
        mismatch_subset = merged_df.copy()

    compiled_dossiers: List[Dict[str, Any]] = []
    
    for _, row in mismatch_subset.iterrows():
        # Double check to ensure we only log genuine priority anomalies
        dossier = build_dossier(row)
        if dossier["mismatch_type"] in ["Hidden Crisis", "False Alarm"]:
            compiled_dossiers.append(dossier)

    # Serialize complete audited collection array to disk layout targets
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(compiled_dossiers, f, indent=2, ensure_ascii=False)

    logger.info(f"Dossier collection compiled successfully. Registered {len(compiled_dossiers)} audits at: {out_file}")
    return compiled_dossiers


if __name__ == "__main__":
    # Integration script runner loop
    generate_dossiers_pipeline()