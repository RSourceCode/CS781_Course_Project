# CS-781 Capstone Phase 1: Legal Language Model Pretraining from Scratch

Phase 1 trains a decoder-only Transformer language model from scratch on Indian legal text. The trained model is later used as the backbone for all downstream phases (NER, rhetorical-role classification, judgment prediction, statute identification, summarization, and precedent retrieval).

## Dataset

All data is obtained from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`.

Training sources (under `Phase_1_Pretraining/Train/`):
- `train_acts.json`
- `train_high_courts.json`
- `train_supreme_court.json`
- `train_tribunals.json`

Test file:
- `Phase_1_Pretraining/Test/test.json` (1000 examples)

Each record is a JSON object with a `"text"` field containing a legal document or excerpt. The four training sources are mixed with equal proportions (0.25 each). Approximately 200 million training tokens and 5 million validation tokens are collected after cleaning.

## Data Cleaning Pipeline

Legal documents (especially OCR-scanned judgments) contain substantial noise that causes repetitive or degenerate generation. A multi-stage cleaning function is applied to every document before tokenization:

1. Unicode normalization (NFKC) to canonicalize homoglyphs, ligatures, and curly quotes.
2. Removal of control characters (keeping only tab, newline, and carriage return) and Unicode format characters (zero-width spaces, soft hyphens, etc.).
3. Repair of OCR hyphenation artifacts of the form “consti-\ntution” → “constitution”.
4. Linearization of pipe-delimited tables into readable “Key: Value” sentences.
5. Removal of long runs of HS tariff codes and similar numeric schedules.
6. Truncation of trailing citation-index / footnote blocks that consist only of parenthesized numbers, year brackets, and initials.
7. Aggressive filtering:
   - Documents whose fraction of “suspicious” characters exceeds 5 % are discarded.
   - Documents shorter than 50 words after cleaning are discarded.
   - Documents containing character-level or word-level repetition loops (or long runs of single-letter initials such as “S.S.S.S.”) are discarded.

Only documents that survive this pipeline are retained for tokenizer training and token collection.

## Tokenizer

A byte-level BPE tokenizer is trained from scratch on the cleaned training text (vocabulary size 20 000). The implementation reuses the Assignment-1 BPE class with the following interface:

- `encode(text)` → list of integer token IDs (always starts with `<s>` and ends with `</s>`)
- `decode(ids)` → reconstructed string
- `get_vocab_size()` / `vocab_size`

Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`. The tokenizer is saved as a pickle file and reused for both training data preparation and generation.

## Model Architecture

The model is a decoder-only transformer whose configuration is controlled by a single `LMConfig` dataclass. The primary variant used for the final submission employs the following design choices (other combinations are supported via flags and were explored experimentally).

**Core dimensions**
- `d_model = 768`
- 12 layers
- 12 query heads, 4 key/value heads (Grouped-Query Attention, GQA)
- Feed-forward dimension 1408 (chosen for SwiGLU compatibility)
- Context length 2048
- Vocabulary size 20 000
- Approximate parameter count of the final model: ~190 M

**Positional encoding**
- Rotary Position Embeddings (RoPE) applied inside the attention layers (preferred).
- Sinusoidal positional encodings added at the input are also supported for ablation.

**Attention variants**
- Standard Causal Self-Attention with GQA and optional RoPE.
- Multi-Head Latent Attention (MLA): KV states are first projected into a low-dimensional latent space (`kv_latent_dim = 192`), then up-projected. This reduces the memory footprint of the KV cache while preserving model quality.

**Feed-forward network**
- Default: standard GELU FFN.
- Preferred: SwiGLU (SiLU-gated linear unit).
- Mixture-of-Experts (MoE) layer: 4 experts, top-2 routing, with an auxiliary load-balancing / z-loss term (coefficient 0.01). When enabled, the FFN of every block becomes an MoE module.

**Normalization and residual structure**
- Pre-LayerNorm (or RMSNorm in the parallel-conv variant) is used throughout; post-norm was found unstable at the chosen learning rate.
- Residual connections follow the standard pre-norm pattern.

**Local Convolution**
- A depth-wise causal convolution (kernel size 25) can be run in parallel with global attention. Learned scalar gates (`alpha` for attention, `beta` for convolution) allow the model to gradually incorporate local context. This variant is controlled by the `use_conv` flag.

All of the above components are implemented from scratch in pure PyTorch; no Hugging Face Transformers model classes are used for the core architecture.

## Training Procedure

Token streams are stored as contiguous `torch.int32` tensors (memory-mapped). Batches are formed by sampling random contiguous windows of length 2048 from the entire mixed token stream, guaranteeing that every batch contains a mixture of Acts, High Court, Supreme Court and Tribunal text.

Training uses:
- AdamW optimizer
- Cosine learning-rate schedule with a short linear warm-up
- Gradient clipping
- Gradient accumulation

Validation loss is evaluated periodically on the held-out 5 M token set. Training and validation loss curves are saved as a PNG plot. The final checkpoint (weights only) is written to disk and can be pushed to the Hugging Face Hub.

Because only a single pass over the data is performed, dropout is set to zero.

## Generation

At inference time the model is placed in evaluation mode and generation proceeds with:

- Temperature 0.7
- Nucleus sampling (`top_p = 0.9`)
- Repetition penalty 1.1
- Special tokens (`<pad>`, `<unk>`, `<s>`, `</s>`) banned from being sampled
- Maximum of 384 new tokens (later truncated to exactly 128 words)

Several practical fixes are applied to the generation routine:
- The trailing `</s>` that the tokenizer always appends is stripped from the prompt so the model continues the current document rather than starting a new one.
- Long prompts are truncated from the left (keeping the most recent tokens) while preserving the leading `<s>`.
- Left-padding is performed together with a proper attention mask so that pad tokens do not participate in attention or shift positional indices.

Generation is performed in batches of 32 for efficiency. The resulting 128-word strings are written to `prediction.json` in the exact format required by Codabench.

## Experimental Setup & Reproducibility

**Hardware.** Experiments were run on Kaggle (2 × Tesla T4) or equivalent CUDA GPUs. The code automatically falls back to CPU if no GPU is present, but training becomes impractically slow.

**Key hyper-parameters (see configuration cell)**
- Vocabulary size: 20 000
- Train tokens: 200 000 000
- Validation tokens: 5 000 000
- Block size / context length: 2048
- Batch size and learning rate are set inside the training loop (typical values: batch size 8–16 with gradient accumulation, peak LR ≈ 3 e-4)

**Running the notebook**

1. Obtain a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone` and store it as a Kaggle secret named `HF`.
2. Execute the cells in order. The first few cells log in, check GPU availability, and set all configuration constants.
3. The data-cleaning and tokenizer-training cells will download the four training files (via `hf_hub_download`) and produce `tokenizerv3.pkl`, `train_tokensv3.pt` and `val_tokensv3.pt`. Subsequent runs simply reload these artifacts.
4. Model definition, training, and checkpointing follow.
5. After training, the test file is downloaded, predictions are generated, and `prediction.json` together with `submission.zip` are written to the working directory.

All random seeds are fixed (`SEED = 42`) for the data-sampling and tokenizer stages. Because of non-deterministic CUDA kernels the exact loss numbers may vary slightly across runs, but the overall training trajectory remains consistent.

## File Layout (produced artifacts)

```
cs781_phase1/
├── tokenizerv3.pkl
├── train_tokensv3.pt
├── val_tokensv3.pt
├── training_validation_loss.png
├── prediction.json
└── submission.zip
```

## Design Rationale & Ablations

The architecture deliberately combines several modern efficiency techniques (GQA, RoPE, SwiGLU, optional MLA and MoE) while remaining simple enough to train from scratch on a few hundred million tokens. Extensive cleaning was found to be more important for generation quality than any single architectural change; models trained on uncleaned data rapidly entered repetition loops. Pre-norm, careful embedding initialization, and the removal of the trailing `</s>` from prompts were the three most impactful engineering fixes that stabilized training and improved continuation quality.
