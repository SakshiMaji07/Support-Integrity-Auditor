# Support Integrity Auditor (SIA)

An Evidence-Grounded Priority Mismatch Detection System for Customer Support Tickets

---

## Overview

Support Integrity Auditor (SIA) is a self-supervised machine learning system designed to detect **Priority Mismatches** in customer support tickets.

A priority mismatch occurs when the objective severity of a support ticket differs from the priority assigned by a human agent. Such inconsistencies can lead to SLA violations, delayed incident response, customer dissatisfaction, and operational inefficiencies.

Unlike traditional supervised classification systems, this project operates under a more challenging setting:

> No ground-truth mismatch labels are available.

Therefore, SIA first generates its own supervision signal through pseudo-labeling and subsequently trains a supervised classifier capable of identifying mismatches in previously unseen tickets.

---

# Problem Statement

Given a support ticket containing:

- Ticket Subject
- Ticket Description
- Ticket Channel
- Customer Domain Information
- Resolution Time
- Human Assigned Priority

the system must:

1. Infer the ticket's true severity independently of the assigned priority.
2. Generate pseudo-labels indicating whether a mismatch exists.
3. Train a classifier using these pseudo-labels.
4. Detect priority mismatches on unseen tickets.
5. Generate evidence-backed explanations for flagged tickets.

---

# Dataset

**Customer Support Tickets – CRM Dataset**

### Key Columns Used

| Column | Purpose |
|----------|----------|
| Ticket Subject | Short issue summary |
| Ticket Description | Full issue description |
| Ticket Priority | Human-assigned label |
| Ticket Channel | Communication channel |
| Resolution Time | Severity proxy |
| Product Purchased | Domain/customer tier proxy |

---

# System Architecture

```text
Raw CRM Tickets
       │
       ▼
Stage 1: Pseudo-Label Generation
       │
       ├── Phi-3 Mini Zero-Shot Severity
       ├── Resolution-Time Regression
       ├── Sentence Transformer Clustering
       │
       ▼
Weighted Severity Fusion
(0.5 / 0.3 / 0.2)
       │
       ▼
Pseudo Severity Label
       │
Compare with Assigned Priority
       │
       ▼
Mismatch Label
       │
       ▼
Stage 2: Fine-Tuned DeBERTa-v3-small
       │
       ▼
Priority Mismatch Predictor
       │
       ▼
Evidence Dossier Generator
```

---

# Stage 1 – Self-Supervised Pseudo-Label Generation

The objective of Stage 1 is to infer a ticket's severity without using the human-assigned priority label.

Three independent signals are combined to generate an inferred severity score.

---

## Signal 1: Phi-3 Mini Zero-Shot Severity Scoring

**Weight: 0.50**

A Phi-3 Mini instruction-tuned language model evaluates the semantic urgency of each ticket using:

- Ticket Subject
- Ticket Description

The model predicts one of four severity levels:

| Severity | Score |
|-----------|--------|
| Low | 1 |
| Medium | 2 |
| High | 3 |
| Critical | 4 |

This signal receives the highest weight because it directly captures the semantic meaning and urgency of the issue.

---

## Signal 2: Resolution-Time Regression

**Weight: 0.30**

Resolution time acts as an indirect severity indicator.

The intuition is that more severe or operationally complex issues generally require longer resolution times.

Resolution times are normalized into a severity score ranging from 1 to 4.

---

## Signal 3: Sentence Transformer Clustering

**Weight: 0.20**

Ticket descriptions are embedded using a Sentence Transformer model.

Semantic clustering groups tickets with similar issue characteristics.

Cluster severity is estimated using:

- Average cluster resolution time
- Average cluster LLM severity
- Cluster urgency patterns

Each ticket inherits the severity tendency of its cluster.

This allows the system to capture latent severity patterns not explicitly stated in ticket text.

---

# Weighted Severity Fusion

The final severity score is computed using:

```text
Fused Severity =
0.50 × LLM Severity
+ 0.30 × Resolution Severity
+ 0.20 × Cluster Severity
```

### Fusion Weights

| Signal | Weight |
|----------|----------|
| Phi-3 Severity | 0.50 |
| Resolution-Time Severity | 0.30 |
| Cluster Severity | 0.20 |

The fused score is mapped back to one of:

- Low
- Medium
- High
- Critical

---

# Pseudo-Label Generation

The inferred severity is compared against the human-assigned priority.

If the disagreement exceeds a predefined threshold:

```python
Mismatch = 1
```

Otherwise:

```python
Mismatch = 0
```

This creates the pseudo-labeled dataset used during Stage 2.

---

# Fusion Justification

The project specification requires combining multiple independent severity signals.

The chosen weighting scheme prioritizes semantic understanding while still leveraging behavioral and structural information:

- Phi-3 Severity (50%) provides direct understanding of issue urgency.
- Resolution Severity (30%) captures operational impact.
- Cluster Severity (20%) captures latent semantic patterns.

This approach reduces dependence on any single signal and improves robustness against noisy or adversarial tickets.

---

# Ablation Study

The following ablation experiments will be reported:

| Configuration | Accuracy | Macro F1 |
|---------------|----------|----------|
| Phi-3 Only | TBD |
| Resolution Only | TBD |
| Clustering Only | TBD |
| Phi-3 + Resolution | TBD |
| Phi-3 + Clustering | TBD |
| Resolution + Clustering | TBD |
| Full Fusion | TBD |

The final submission will replace TBD values with experimental results.

---

# Stage 2 – Fine-Tuned Priority Mismatch Classifier

The pseudo-labeled dataset generated in Stage 1 is used to train a supervised binary classifier.

## Model

**DeBERTa-v3-small**

Reasons for selection:

- Strong NLP performance
- Lightweight architecture
- Fast fine-tuning
- Suitable for limited hardware environments

---

## Input Features

### Text Features

- Ticket Subject
- Ticket Description

### Metadata Features

- Ticket Channel
- Domain Tier

### Stage 1 Severity Features

- LLM Severity Score
- Resolution Severity Score
- Cluster Severity Score
- Fused Severity Score

These additional severity signals provide valuable context beyond raw text.

---

## Output Classes

| Label | Meaning |
|---------|----------|
| 0 | Consistent |
| 1 | Priority Mismatch |

---

## Handling Class Imbalance

Priority mismatches are expected to be significantly less common than correctly labeled tickets.

To address this imbalance, the model uses:

### Weighted Cross-Entropy Loss

Class weights are computed using label frequencies in the pseudo-labeled training set.

This increases the penalty for misclassifying minority-class mismatch examples and improves recall.

---

# Evidence Dossier Generation

For every ticket predicted as a mismatch, the system generates an evidence-grounded dossier.

Example:

```json
{
  "ticket_id": "1234",
  "assigned_priority": "Low",
  "inferred_severity": "High",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": "+2",
  "feature_evidence": [
    {
      "signal": "phi3_severity",
      "value": "High",
      "weight": "0.50"
    },
    {
      "signal": "resolution_time",
      "value": "72 hours",
      "interpretation": "Above average"
    }
  ],
  "constraint_analysis":
  "Ticket language indicates service disruption while assigned priority remains Low.",
  "confidence": "0.91"
}
```

All evidence items are derived directly from ticket fields or computed severity signals.

No fabricated or unverifiable information is introduced.

---

# Streamlit Dashboard

The deployed web application supports:

## Single Ticket Analysis

Users can submit an individual ticket and receive:

- Mismatch prediction
- Confidence score
- Evidence dossier

## Batch CSV Processing

Users can upload CSV files containing multiple tickets and receive:

- Predictions
- Severity scores
- Evidence reports

## Analytics Dashboard

Visualizations include:

- Mismatch distribution
- Hidden Crisis vs False Alarm counts
- Top contributing signals
- Severity delta heatmaps
- Channel-wise mismatch analysis

---

# Repository Structure

```text
Support-Integrity-Auditor/
│
├── notebook.ipynb
├── requirements.txt
├── README.md
├── predict.py
│
├── data/
│      │──processed/
│      │──raw/
├── models/
├── outputs/
├── src/
│      │── dossier_generator.py
│      │── preprocessing.py
│      │── pseudo_labels.py
│      │── train_model.py
├── app.py
├── train_pipeline.py
```

---

# Technology Stack

### Machine Learning

- PyTorch
- Hugging Face Transformers
- DeBERTa-v3-small
- Phi-3 Mini
- Sentence Transformers
- Scikit-Learn

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Deployment

- Streamlit

---

# Future Improvements

- Learn fusion weights automatically instead of manually assigning them.
- Incorporate graph-based ticket relationships.
- Introduce active learning for pseudo-label refinement.
- Improve adversarial robustness.
- Add Retrieval-Augmented Evidence Generation (RAG).

---
