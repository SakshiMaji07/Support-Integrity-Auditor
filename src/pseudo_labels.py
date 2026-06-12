"""
Pseudo-label generation pipeline for Support Integrity Auditor.

Generates inferred ticket severity without using the ticket priority column by combining:
1. LLM severity signal (using phi-3)
2. Resolution-time severity signal
3. Cluster severity signal (using sentence transformers + semantic keyword profiles)
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging as transformers_logging,
    pipeline,
)

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)

# Priority to Severity Mapping
PRIORITY_TO_SEVERITY = {
    "P1": 4,
    "P2": 3,
    "P3": 2,
    "P4": 1,
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}

# Severity level names mapping for strings
SEVERITY_LEVELS = {
    0.25: "LOW",
    0.5: "MEDIUM",
    0.75: "HIGH",
    1.0: "CRITICAL",
}

# String fallback mapping for logging metrics safely
SEVERITY_LEVEL_NAMES = {
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
    4: "CRITICAL",
}

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

_phi3_pipe = None


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _load_llm_pipeline() -> None:
    global _phi3_pipe
    if _phi3_pipe is not None:
        return

    auth_token = _get_hf_token()
    auth_kwargs = {"token": auth_token} if auth_token else {}
    
    # Debug: Log token presence
    if auth_token:
        token_preview = auth_token[:10] + "..." if len(auth_token) > 10 else auth_token
        logger.info("Using HF token: %s", token_preview)
    else:
        logger.warning("No HF_TOKEN found in environment variables")
    
    cuda_available = torch.cuda.is_available()
    model_kwargs = {"torch_dtype": torch.float16 if cuda_available else torch.float32}
    
    if _has_accelerate() and cuda_available:
        model_kwargs["device_map"] = "auto"

    def _load_from_hub(local_only: bool) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=local_only,
            trust_remote_code=True,
            **auth_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            local_files_only=local_only,
            trust_remote_code=True,
            **model_kwargs,
            **auth_kwargs,
        )
        return tokenizer, model

    try:
        logger.info("Attempting to load LLM from local cache...")
        tokenizer, model = _load_from_hub(local_only=True)
        logger.info("Loaded LLM from local cache.")
    except Exception as local_exc:
        logger.info("Local cache load failed (expected if first time): %s", str(local_exc)[:100])
        try:
            logger.info("Attempting to load LLM from HuggingFace Hub...")
            tokenizer, model = _load_from_hub(local_only=False)
            logger.info("Loaded LLM from HF Hub.")
        except Exception as hub_exc:
            logger.error(
                "Unable to load HF model for LLM severity inference.\n"
                "Error type: %s\n"
                "Error message: %s\n"
                "Falling back to keyword-based severity.",
                type(hub_exc).__name__,
                str(hub_exc),
            )
            _phi3_pipe = None
            return

    if "device_map" not in model_kwargs:
        device = 0 if cuda_available else -1
    else:
        device = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    _phi3_pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=device,
        batch_size=32
    )


def _keyword_severity_fallback(text: str) -> float:
    if pd.isna(text):
        return 0.25

    text_lower = str(text).lower()
    if any(term in text_lower for term in [
        "outage", "production down", "service unavailable",
        "security breach", "data corruption", "critical", "emergency",
    ]):
        return 1.0
    if any(term in text_lower for term in [
        "major issue", "feature broken", "business impact",
        "failed", "error", "failure",
    ]):
        return 0.75
    if any(term in text_lower for term in [
        "disruption", "workaround", "slow", "performance",
        "limited", "partial",
    ]):
        return 0.5
    return 0.25


SEVERITY_PROMPT = """You are a support operations analyst.

Classify the ticket severity:
CRITICAL
HIGH
MEDIUM
LOW

Definitions:
CRITICAL: Production outage, security incident, data corruption, or large-scale failure.
HIGH: Major feature broken, significant business impact.
MEDIUM: Workflow disruption, standard support issue.
LOW: Cosmetic issue, question, feature request.

Return ONLY one word.

Ticket: {ticket}
Severity:"""


def load_processed_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load the processed CSV data."""
    if file_path is None:
        project_root = Path(__file__).resolve().parents[1]
        file_path = project_root / "data" / "processed" / "processed.csv"

    file_path = Path(file_path)
    logger.info("Loading processed data from %s", file_path)

    if not file_path.exists():
        logger.error("File does not exist: %s", file_path)
        raise FileNotFoundError(f"File does not exist: {file_path}")

    df = pd.read_csv(file_path)
    logger.info("Loaded %d tickets", len(df))
    return df


def get_llm_severity_batch(texts: pd.Series) -> list[float]:
    """Processes entire Series through GPU-accelerated pipeline natively via batched iteration."""
    _load_llm_pipeline()
    
    if _phi3_pipe is None:
        return [_keyword_severity_fallback(t) for t in texts]

    scores = []
    prompts = [SEVERITY_PROMPT.format(ticket=str(t)[:2000]) if pd.notna(t) else "" for t in texts]
    
    try:
        results = _phi3_pipe(
            prompts,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
            truncation=True
        )
        
        for idx, result in enumerate(results):
            if prompts[idx] == "":
                scores.append(0.25)
                continue
                
            generated = result[0]["generated_text"].upper()
            assigned = False
            for val, label in SEVERITY_LEVELS.items():
                if label in generated:
                    scores.append(val)
                    assigned = True
                    break
            if not assigned:
                scores.append(_keyword_severity_fallback(texts.iloc[idx]))
                
    except Exception as exc:
        logger.warning("Pipeline batch processing failed. Running safe baseline fallbacks instead: %s", exc)
        return [_keyword_severity_fallback(t) for t in texts]

    return scores


def generate_embeddings(texts: pd.Series) -> np.ndarray:
    """Generate sentence embeddings using sentence-transformers or TF-IDF backup."""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            texts.astype(str).tolist(),
            show_progress_bar=True,
            batch_size=32
        )
        logger.info("Generated embeddings with shape %s", embeddings.shape)
        return np.asarray(embeddings)
    except ImportError:
        logger.warning(
            "sentence-transformers not available, using simple feature-based approach"
        )
        vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        return vectorizer.fit_transform(texts.astype(str)).toarray()


def generate_cluster_profiles(
    texts: pd.Series,
    cluster_labels: np.ndarray,
    n_clusters: int = 5,
    top_n: int = 5,  # Increased default slightly for better semantic analysis mapping
) -> dict[int, str]:
    """Generate human-readable profile for each cluster using TF-IDF."""
    profiles = {}
    try:
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_texts = texts[cluster_mask].astype(str).tolist()
            
            if not cluster_texts:
                profiles[cluster_id] = ""
                continue
            
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                top_indices = avg_tfidf.argsort()[-top_n:][::-1]
                top_terms = [feature_names[i] for i in top_indices if avg_tfidf[i] > 0]
                profiles[cluster_id] = ",".join(top_terms) if top_terms else ""
            except ValueError:
                profiles[cluster_id] = ""
    except Exception as e:
        logger.warning("Error generating cluster profiles: %s", e)
        for cluster_id in range(n_clusters):
            profiles[cluster_id] = ""
    
    return profiles


def perform_clustering(
    embeddings: np.ndarray,
    texts: pd.Series,
    n_clusters: int = 5,
) -> tuple[np.ndarray, dict[int, float]]:
    """Perform K-means clustering and calculate cluster severities based on cluster semantics."""
    logger.info("Performing K-means clustering with %d clusters...", n_clusters)
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    logger.info("Clustering complete. Cluster distribution: %s", np.bincount(cluster_labels))
    
    # Generate profiles context explicitly to deduce semantics
    cluster_profiles = generate_cluster_profiles(texts, cluster_labels, n_clusters=n_clusters)
    
    # Map high/low signal keywords to score targets
    semantic_weights = {
        "outage": 1.0, "down": 1.0, "unavailable": 1.0, "breach": 1.0, "corruption": 1.0, "critical": 1.0, "emergency": 1.0, "crash": 1.0,
        "broken": 0.75, "failed": 0.75, "error": 0.75, "failure": 0.75, "impact": 0.75, "major": 0.75,
        "disruption": 0.50, "workaround": 0.50, "slow": 0.50, "performance": 0.50, "limited": 0.50, "partial": 0.50,
        "question": 0.25, "request": 0.25, "cosmetic": 0.25, "ui": 0.25, "documentation": 0.25
    }
    
    cluster_severity_map = {}
    for cluster_id in range(n_clusters):
        profile_text = cluster_profiles.get(cluster_id, "").lower()
        matched_scores = []
        
        for word, weight in semantic_weights.items():
            if word in profile_text:
                matched_scores.append(weight)
                
        # If keywords match semantic targets, calculate the average, else fall back to a safe baseline median
        if matched_scores:
            cluster_severity_map[cluster_id] = float(np.mean(matched_scores))
        else:
            # Fallback baseline when cluster contains neutral terminology
            cluster_severity_map[cluster_id] = 0.50
            
        logger.info("Cluster %d Semantics: [%s] -> Assigned Severity Score: %.2f", 
                    cluster_id, cluster_profiles.get(cluster_id, "None"), cluster_severity_map[cluster_id])
        
    return cluster_labels, cluster_severity_map


def fuse_signals(
    llm_score: pd.Series | np.ndarray,
    resolution_score: pd.Series | np.ndarray,
    cluster_score: pd.Series | np.ndarray,
    llm_weight: float = 0.5,
    resolution_weight: float = 0.3,
    cluster_weight: float = 0.2,
) -> np.ndarray:
    """Fuse three severity signals into a final score vectorially."""
    total_weight = llm_weight + resolution_weight + cluster_weight
    
    fused = (
        llm_score * (llm_weight / total_weight)
        + resolution_score * (resolution_weight / total_weight)
        + cluster_score * (cluster_weight / total_weight)
    )
    return np.array(fused)


def extract_trigger_keywords(text: str) -> str:
    """Extract trigger keywords from ticket text that match severity indicators."""
    if pd.isna(text):
        return ""
    
    critical_keywords = [
        r'\boutage\b', r'\bproduction\s*down\b', r'\bservice\s*unavailable\b',
        r'\bsecurity\s*breach\b', r'\bdata\s*corruption\b', r'\bcritical\b',
        r'\bemergency\b', r'\bcrash\b', r'\bdown\b'
    ]
    high_keywords = [
        r'\bfeature\s*broken\b', r'\bmajor\s*issue\b', r'\bbusiness\s*impact\b',
        r'\bsignificant\b', r'\bfailed\b', r'\berror\b', r'\bfailure\b'
    ]
    medium_keywords = [
        r'\bdisruption\b', r'\bworkaround\b', r'\bslow\b', r'\bperformance\b',
        r'\blimited\b', r'\bpartial\b'
    ]
    
    matched = []
    text_lower = str(text).lower()
    
    for pattern in critical_keywords:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(pattern.replace(r'\b', '').replace(r'\s*', ' ').strip('^$'))
    for pattern in high_keywords:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(pattern.replace(r'\b', '').replace(r'\s*', ' ').strip('^$'))
    for pattern in medium_keywords:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(pattern.replace(r'\b', '').replace(r'\s*', ' ').strip('^$'))
            
    seen = set()
    unique_matched = []
    for keyword in matched:
        if keyword not in seen:
            seen.add(keyword)
            unique_matched.append(keyword)
    
    return "|".join(unique_matched) if unique_matched else ""


def generate_evidence_summary(
    ticket_id: str,
    trigger_keywords: str,
    cluster_profile: str,
    resolution_interpretation: str,
    inferred_severity: int,
) -> str:
    """Generate concise grounded explanation using available evidence signals."""
    summary_parts = []
    
    if trigger_keywords:
        keyword_list = trigger_keywords.split("|")[:3]
        summary_parts.append(
            f"Ticket contains {len(keyword_list)} high-priority keywords ({', '.join(keyword_list)})."
        )
    if cluster_profile:
        summary_parts.append(
            f"Cluster analysis shows this ticket is grouped with issues related to {cluster_profile}."
        )
    if resolution_interpretation:
        summary_parts.append(resolution_interpretation)
    
    if summary_parts:
        combined = " ".join(summary_parts[:3])
        return combined if len(combined) <= 300 else combined[:297] + "..."
    else:
        severity_name = SEVERITY_LEVEL_NAMES.get(inferred_severity, "Unknown")
        return f"Ticket has been inferred as {severity_name} severity based on available signals."


def generate_and_save_evidence(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate evidence DataFrame and save to CSV."""
    logger.info("Generating evidence for Stage 3 dossier generation...")
    
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "data" / "processed"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_clusters = len(np.unique(cluster_labels))
    cluster_profiles = generate_cluster_profiles(df["combined_text"], cluster_labels, n_clusters=n_clusters)
    
    res_times_series = df["Resolution_Time_Hours"]
    p25, p50, p75, p90 = res_times_series.quantile([0.25, 0.50, 0.75, 0.90])
    
    interpretations = np.where(res_times_series.isna(), "Resolution time information not available.",
                      np.where(res_times_series <= p25, "Resolution time is well below expected range for standard incidents.",
                      np.where(res_times_series <= p50, "Resolution time is consistent with typical low-to-medium priority tickets.",
                      np.where(res_times_series <= p75, "Resolution time exceeds the median but remains within normal bounds.",
                      np.where(res_times_series <= p90, "Resolution time is notably extended, suggesting complex issues or high priority.",
                                                               "Resolution time is substantially extended, indicating prolonged operational impact.")))))

    evidence_data = {
        "Ticket_ID": df["Ticket_ID"].tolist(),
        "Trigger_Keywords": [extract_trigger_keywords(text) for text in df["combined_text"]],
        "Cluster_ID": cluster_labels.astype(int).tolist(),
        "Cluster_Profile": [cluster_profiles.get(int(cid), "") for cid in cluster_labels],
        "Resolution_Interpretation": interpretations.tolist(),
        "Evidence_Summary": []
    }
    
    for idx in range(len(df)):
        evidence_data["Evidence_Summary"].append(
            generate_evidence_summary(
                str(evidence_data["Ticket_ID"][idx]),
                evidence_data["Trigger_Keywords"][idx],
                evidence_data["Cluster_Profile"][idx],
                evidence_data["Resolution_Interpretation"][idx],
                int(df["Inferred_Severity"].iloc[idx]),
            )
        )
        
    evidence_df = pd.DataFrame(evidence_data)
    evidence_path = output_dir / "evidence.csv"
    evidence_df.to_csv(evidence_path, index=False)
    
    logger.info("Saved evidence to %s", evidence_path)
    return evidence_df


def generate_pseudo_labels(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate pseudo-labels using fully vectorized workflows."""
    df = load_processed_data(input_path)
    logger.info("Processing %d tickets", len(df))
    
    logger.info("Generating embeddings...")
    embeddings = generate_embeddings(df["combined_text"])
    
    # CHANGED: Passed df["combined_text"] to perform semantic keyword tracking during cluster setup
    cluster_labels, cluster_severity_map = perform_clustering(embeddings, df["combined_text"])

    # 1. Vectorized Resolution Time Quantiles Calculation
    res_times = df["Resolution_Time_Hours"]
    p25, p50, p75, p90 = res_times.quantile([0.25, 0.50, 0.75, 0.90])
    
    df["Resolution_Severity"] = np.where(res_times.isna(), 0.50,
                                np.where(res_times <= p25, 0.25,
                                np.where(res_times <= p50, 0.50,
                                np.where(res_times <= p75, 0.70,
                                np.where(res_times <= p90, 0.85, 0.95)))))

    # 2. Corrected Cluster Severity Mapping (Uses actual generated semantic K-Means labels array)
    df["Cluster_Severity"] = pd.Series(cluster_labels).map(cluster_severity_map).fillna(0.5).values

    # 3. Optimized Batched GPU Tokenization Pipeline Mapping
    logger.info("Running batched GPU inference for LLM Severities...")
    df["LLM_Severity"] = get_llm_severity_batch(df["combined_text"])
    
    # 4. Vectorized Signal Fusion
    df["Fused_Severity"] = fuse_signals(df["LLM_Severity"], df["Resolution_Severity"], df["Cluster_Severity"])
    
    # 5. Vectorized Cutoff Conversion to Levels Boundaries
    fused = df["Fused_Severity"]
    df["Inferred_Severity"] = np.where(fused < 0.3, 1,
                              np.where(fused < 0.55, 2,
                              np.where(fused < 0.75, 3, 4)))
    
    # 6. Vectorized Real Priority Mappings
    df["Assigned_Severity"] = df["Priority_Level"].astype(str).str.strip().str.upper().map(PRIORITY_TO_SEVERITY).fillna(2).astype(int)
    
    # 7. Mismatch Flag logic 
    df["Mismatch_Label"] = np.where(np.abs(df["Inferred_Severity"] - df["Assigned_Severity"]) >= 2, 1, 0)
    
    output_columns = [
        "Ticket_ID",
        "Priority_Level",
        "Assigned_Severity",
        "LLM_Severity",
        "Resolution_Severity",
        "Cluster_Severity",
        "Fused_Severity",
        "Inferred_Severity",
        "Mismatch_Label",
    ]
    output_df = df[output_columns].copy()
    
    if output_path is None:
        project_root = Path(__file__).resolve().parents[1]
        output_path = project_root / "data" / "processed" / "pseudo_labeled_tickets.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    logger.info("Saved pseudo-labeled tickets to %s", output_path)
    
    logger.info("Inferred severity distribution:")
    for level in [1, 2, 3, 4]:
        count = (output_df["Inferred_Severity"] == level).sum()
        pct = (count / len(output_df)) * 100
        logger.info(
            "  Level %d (%s): %d tickets (%.1f%%)",
            level,
            SEVERITY_LEVEL_NAMES[level],
            count,
            pct,
        )
        
    generate_and_save_evidence(df, cluster_labels, output_path.parent)
    return output_df


def main() -> None:
    """Main entry point for pseudo-label generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting Pseudo-Label Generation Pipeline")
    
    try:
        output_df = generate_pseudo_labels()
        logger.info("Pipeline completed successfully!")
        logger.info("Output shape: %s", output_df.shape)
    except Exception as e:
        logger.error("Pipeline failed with error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
