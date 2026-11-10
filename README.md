# CS-781 Capstone Project  
**Legal Language Modelling**

**Author:** Rahul Singh  
**Roll Number:** 230834  

---

## Project Overview

This repository contains the complete solution for the **CS-781 Capstone Project**.  
The work is organised into **seven sequential phases**:

| Phase | Task | Short Description |
|-------|------|-------------------|
| **1** | Legal Language Model Pretraining | Train a decoder-only Transformer from scratch on Indian legal text |
| **2** | Legal Named Entity Recognition (LNER) | Token-level BIO tagging of entities in legal documents |
| **3** | Rhetorical Role Classification | Sentence-level classification of rhetorical roles (Fact, Statute, Ratio, …) |
| **4** | Court Judgment Prediction (CJPE) | Binary outcome prediction for long court cases |
| **5** | Legal Statute Identification (LSI) | Multi-label prediction of relevant statutes for a case |
| **6** | Legal Document Summarization | Abstractive summarization of long judgments |
| **7** | Precedent Case Retrieval (PCR) | Dense retrieval of relevant prior cases |

All downstream phases (2–7) reuse the same family of Phase-1 pretrained backbones and the same custom Byte-Level BPE tokenizer.

---

## Repository Structure

```
.
├── phase1/
│   ├── Phase1.ipynb
│   └── ph1.md
├── phase2/
│   ├── Phase2.ipynb
│   └── ph2.md
├── phase3/
│   ├── Phase3.ipynb
│   └── ph3.md
├── phase4/
│   ├── Phase4.ipynb
│   └── ph4.md
├── phase5/
│   ├── Phase5.ipynb
│   └── ph5.md
├── phase6/
│   ├── Phase6.ipynb
│   └── ph6.md
└── phase7/
    ├── Phase7.ipynb
    └── ph7.md
```

Each `phX.md` file contains a detailed technical README for that phase (architecture, methodology, experimental setup, and how to run the notebook).

---

## Shared Infrastructure

### Tokenizer
- Custom **Byte-Level BPE** trained from scratch on the cleaned legal corpus.
- Vocabulary size: **20 000**
- Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`
- Saved as `tokenizerv3.pkl` and reused by every phase.

### Phase-1 Model Family
Four decoder-only Transformer variants were pretrained:

| Name                       | Layers | Attention | Position   | FFN              | Extra                |
|----------------------------|--------|-----------|------------|------------------|----------------------|
| `gqa-rope`                 | 8      | GQA       | RoPE       | Standard         | –                    |
| `mla-sin`                  | 12     | MLA       | Sinusoidal | Standard         | –                    |
| `gqa-rope-swiglu-moe`      | 12     | GQA       | RoPE       | SwiGLU + MoE     | –                    |
| `swiglu-moe-rope-conv`     | 12     | GQA       | RoPE       | SwiGLU + MoE     | Parallel local conv  |

- `d_model = 768`, maximum sequence length = 2 048  
- Main model used for most downstream experiments: **`gqa-rope-swiglu-moe`**

### Common Design Patterns Across Downstream Phases
- Parameter-efficient fine-tuning (LoRA, LoRMA, AdaLoRA, Adapters, BitFit, Prompt/Prefix, IA³, …)
- Knowledge distillation (full teacher → half-depth student)
- Long-document handling via overlapping / selected chunks
- Gradient checkpointing and mixed-precision training
- Fully checkpointed, resume-safe experiment loops
- Codabench-ready `prediction.json` + `submission.zip` generation

---

## Phase Summaries

### Phase 1 – Pretraining
Causal language modeling on 200 M tokens of cleaned Indian legal text (Acts, High Courts, Supreme Court, Tribunals).  
Aggressive document cleaning pipeline for OCR noise, tables, repetition loops, etc.  
Final model is used as the backbone for all subsequent phases.

### Phase 2 – Legal Named Entity Recognition
Token-level BIO tagging on long documents.  
Overlapping windows + Viterbi decoding.  
Extensive comparison of PEFT methods, loss functions (CE / weighted CE / focal), and causal vs bidirectional attention.

### Phase 3 – Rhetorical Role Classification
Sentence-level multi-class classification (Fact, Statute, Ratio of the Decision, Arguments, Precedents, …) on two domains ( Corporate Law &  Information Technology).  
Context-aware sentence encoding and the same PEFT / KD toolkit.

### Phase 4 – Court Judgment Prediction
Binary classification of case outcome.  
Head-Tail chunk encoder + mean / attention / BiGRU pooling.  
Threshold tuning on the development set and ensemble support.

### Phase 5 – Legal Statute Identification
Multi-label prediction of relevant statutes.  
Severe class imbalance handled by weighted BCE, focal loss, positive-weight capping.  
Global / per-label threshold tuning.

### Phase 6 – Legal Document Summarization
Abstractive summarization in a causal LM setup (`document + “Summary:” + summary`).  
Long-document packing strategies (head-tail, uniform, hierarchical summary-of-summaries).  
Evaluation with ROUGE-1/L and BERTScore.

### Phase 7 – Precedent Case Retrieval
Dense retrieval of relevant prior cases.  
Dual-encoder architecture with cosine or MaxSim similarity.  
InfoNCE / Triplet losses, TF-IDF hard-negative mining, and ranking metrics (F1@7, R@20, R@50).

---

## How to Use This Repository

1. **Environment**
   - Kaggle notebooks or any CUDA-enabled machine with PyTorch ≥ 2.0.
   - Additional packages used in later phases: `ijson`, `rouge-score`, `bert-score`.

2. **Hugging Face Authentication**
   - Store the HF token as a Kaggle secret named `HF` (or set the environment variable).

3. **Phase-1 Artifacts**
   - After running Phase 1, keep:
     - `tokenizerv3.pkl`
     - the four model checkpoints (`gqa-rope.pt`, `mla-sin.pt`, `gqa-rope-swiglu-moe.pt`, `swiglu-moe-rope-conv.pt`)
   - These files are required by Phases 2–7.

4. **Running a Phase**
   - Open the corresponding notebook.
   - Read the phase-specific `phX.md` for configuration options, experiment lists, and expected outputs.
   - Set `RUN_MODE` / `SMOKE_TEST` as needed (full / ablation / smoke).
   - Execute cells in order; finished experiments are automatically skipped on re-runs.

5. **Submissions**
   - Each phase writes one or more `submission_*.zip` files containing `prediction.json`.

---

*For detailed technical documentation of any individual phase, please refer to the corresponding `phX.md` file inside each phase directory.*
