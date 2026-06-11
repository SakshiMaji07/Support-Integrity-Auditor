from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "Ticket_ID",
    "Ticket_Subject",
    "Ticket_Description",
    "Priority_Level",
    "Resolution_Time_Hours",
    "Ticket_Channel",
    "Issue_Category",
]

URGENT_KEYWORDS: List[str] = [
    "urgent",
    "asap",
    "immediately",
    "critical",
    "now",
    "emergency",
]

ESCALATION_PHRASES: List[str] = [
    "escalate",
    "escalation",
    "speak to manager",
    "higher level",
    "supervisor",
    "complaint",
    "not satisfied",
    "disappointed",
    "unacceptable",
]

NEGATION_WORDS: List[str] = [
    "not",
    "cannot",
    "cant",
    "can't",
    "never",
    "failed",
    "no",
    "none",
]


def load_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load customer support ticket data from the raw data folder."""
    if file_path is None:
        project_root = Path(__file__).resolve().parents[1]
        file_path = project_root / "data" / "raw" / "customer_support_tickets.csv"

    file_path = Path(file_path)
    logger.info("Loading ticket data from %s", file_path)

    if not file_path.exists():
        logger.error("Ticket file does not exist: %s", file_path)
        raise FileNotFoundError(f"Ticket file does not exist: {file_path}")

    df = pd.read_csv(file_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        logger.error("Missing required columns: %s", missing_columns)
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def clean_text(text: object) -> str:
    """Normalize text by collapsing whitespace while preserving negations."""
    if pd.isna(text):
        return ""

    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized


def _count_pattern_matches(text: str, patterns: List[str]) -> int:
    count = 0
    for pattern in patterns:
        escaped_pattern = re.escape(pattern)
        regex = re.compile(rf"\b{escaped_pattern}\b", flags=re.IGNORECASE)
        count += len(regex.findall(text))
    return count


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rule-based features to the ticket DataFrame."""
    df = df.copy()

    df["urgent_keyword_count"] = df["combined_text"].astype(str).apply(
        lambda value: _count_pattern_matches(value, URGENT_KEYWORDS)
    )
    df["escalation_phrase_count"] = df["combined_text"].astype(str).apply(
        lambda value: _count_pattern_matches(value, ESCALATION_PHRASES)
    )
    df["negation_count"] = df["combined_text"].astype(str).apply(
        lambda value: _count_pattern_matches(value, NEGATION_WORDS)
    )

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and enrich ticket data for model input."""
    df = df.copy()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        logger.error("Missing required columns in DataFrame: %s", missing_columns)
        raise ValueError(f"Missing required columns in DataFrame: {missing_columns}")

    df["Ticket_Subject"] = df["Ticket_Subject"].fillna("").astype(str).apply(clean_text)
    df["Ticket_Description"] = df["Ticket_Description"].fillna("").astype(str).apply(clean_text)

    df = df[df["Priority_Level"].notna() & df["Priority_Level"].astype(str).str.strip().ne("")].copy()
    logger.info("Kept %d rows after dropping missing priority values", len(df))

    df["combined_text"] = df["Ticket_Subject"] + " [SEP] " + df["Ticket_Description"]
    df = create_features(df)

    return df


def save_processed_data(df: pd.DataFrame, file_path: str | Path | None = None) -> Path:
    """Save the processed ticket data to the processed data folder."""
    if file_path is None:
        project_root = Path(__file__).resolve().parents[1]
        file_path = project_root / "data" / "processed" / "processed.csv"

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(file_path, index=False)
    logger.info("Saved processed data to %s", file_path)
    return file_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        df = load_data()
        clean_df = preprocess_dataframe(df)
        save_processed_data(clean_df)
        logger.info("Preprocessing complete. Processed %d tickets.", len(clean_df))
    except Exception as error:
        logger.exception("Failed to preprocess ticket data: %s", error)
        raise


if __name__ == "__main__":
    main()
