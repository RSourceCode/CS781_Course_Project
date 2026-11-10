# CS-781 Capstone Phase 3: Rhetorical Role Classification (RR)

This notebook implements **Phase 3** of the CS-781 Capstone Project: fine-tuning the Phase-1 pretrained legal language models for **Rhetorical Role (RR) classification** of sentences in Indian legal judgments.

Each sentence in a judgment is assigned one of a fixed set of rhetorical roles (Fact, Statute, Ratio of the Decision, Argument by Petitioner / Respondent, various Precedent categories, Ruling by Present / Lower Court, Issue, Dissent, etc.). The task is multi-class sentence classification performed independently for two legal domains:

- **CL** – Corporate Law
- **IT** – Information Technology

A classification head is placed on top of a Phase-1 decoder backbone. Multiple parameter-efficient fine-tuning (PEFT) techniques, loss functions, context representations, and knowledge-distillation variants are systematically compared. The best models (across techniques and architectures) are used to produce the final Codabench submissions.

## Task Formulation and Data

Data live under `Phase_3_RR` in the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. Each example is a judgment that has already been split into sentences; every sentence carries a gold rhetorical-role label. The label inventory is derived from the training data and is shared across both domains. Sentences that serve only as context (or that fall outside the labelled span) receive the ignore index `-100` and are excluded from the loss.

Because a judgment may contain dozens of sentences, each sentence is encoded independently up to a hard limit of `MAX_SENT_TOK = 160` tokens (an `</s>` token is appended). The resulting per-sentence representations are then combined by a lightweight sequence model that sees the whole judgment, allowing the classifier to exploit inter-sentence context while still fitting inside GPU memory.

Three run modes control data volume and wall-clock budget:
- `full` – complete data and the intended number of epochs (production runs and submissions);
- `ablation` – seeded 30 % / 50 % slices of train/dev with a hard epoch cap (fast method comparison);
- `smoke` – a handful of documents per domain and a single epoch (end-to-end pipeline check).

## Phase-1 Backbone Architectures

Four pretrained checkpoints are available and are loaded by name from `PHASE1_CONFIGS`:

- `gqa-rope` – 8 layers, standard GELU FFN, grouped-query attention + RoPE;
- `mla-sin` – 12 layers, multi-head latent attention, sinusoidal positional encodings;
- `gqa-rope-swiglu-moe` – 12 layers, SwiGLU + Mixture-of-Experts (4 experts, top-2), GQA + RoPE (default backbone for technique experiments);
- `swiglu-moe-rope-conv` – same as the MoE model plus a parallel depth-wise causal convolution (kernel 25) inside each block.

All backbones share `d_model = 768`, 12 query heads (4 KV heads for GQA variants) and a maximum sequence length of 2048. The Phase-1 `DecoderLM` implementation (RoPE, MLA, MoE, optional local convolution) is reused unchanged; only the language-model head is replaced by the rhetorical-role classification head.

## Sentence Encoding and Document-Level Modelling

For every sentence the Phase-1 backbone produces a sequence of hidden states. A mean-pool (or CLS-style) aggregation yields a fixed-size sentence embedding. These embeddings are stacked in document order and fed to a small bidirectional transformer or LSTM that models inter-sentence dependencies. The final per-sentence hidden states are projected by a linear classifier onto the rhetorical-role vocabulary. Dropout is applied before the classifier.

The same adaptation methods developed in Phase 2 are available: full fine-tuning, partial unfreezing of the top blocks, gradual unfreezing, BitFit, LoRA, LoRMA, AdaLoRA and bottleneck adapters. Knowledge distillation from a fully fine-tuned teacher onto a thinner student (every second block kept) is also supported. The attention mask inside the backbone can be left causal or made bidirectional; both settings are ablated.

## Loss Functions and Class Imbalance

Rhetorical roles are heavily skewed. Three loss variants are implemented:

- ordinary cross-entropy;
- inverse-frequency weighted cross-entropy;
- focal loss (focusing parameter `FOCAL_GAMMA = 2.0`) that further down-weights easy majority-class examples while still using class weights.

When the backbone contains a Mixture-of-Experts layer the router load-balancing (z-loss) term can optionally be added to the task loss (`MOE_AUX`).

## Training Loop, Checkpointing and Resource Control

Training uses AdamW with separate learning rates for backbone/adapter parameters and for the classification head, a short linear warm-up followed by cosine decay that never falls below a configurable floor (`MIN_LR_FRAC`), gradient clipping, and optional gradient checkpointing. Mixed precision (bfloat16 when available, otherwise float16) is enabled on CUDA. A hard wall-clock limit (`MAX_MIN_PER_RUN`) aborts any experiment that exceeds the allotted minutes, guaranteeing that long ablation sweeps stay inside the available compute budget.

Every epoch the model is evaluated on the development set of each domain with macro-F1. The best checkpoint (by average macro-F1 across domains) and a full resume checkpoint (model, optimiser, scaler, AdaLoRA importance scores, RNG state) are written atomically, so training can be interrupted and resumed without loss of reproducibility. After the final epoch the best weights are restored before test-set inference.

## Experimental Protocol

A typical campaign proceeds as follows.

1. Full bidirectional fine-tuning of the main backbone yields both the strongest single model and the teacher for distillation.
2. A thinner student is trained with and without distillation from that teacher.
3. The suite of parameter-efficient methods (LoRA, LoRMA, AdaLoRA, adapters, BitFit, partial and gradual unfreezing) is run on the same backbone.
4. The strongest PEFT method is repeated on every Phase-1 architecture to measure backbone sensitivity.
5. Controlled ablations examine causal versus bidirectional attention, different loss formulations, and the contribution of the document-level sequence model.

All runs share the same random seed, the same data splits (under a given `RUN_MODE`) and the same evaluation code, so observed differences are attributable only to the experimental factor under study. Results are serialised to JSON; helper routines aggregate them into a ranked table, produce per-role and confusion-matrix analyses, and write the required Codabench submission zips (four distinct techniques plus one best-per-architecture archive).

## Running the Code

1. Place the Phase-1 tokenizer (`tokenizerv3.pkl`) and the four model checkpoints into the working directory (or adjust the copy commands at the top of the notebook).
2. Authenticate with a Hugging Face token that can read `Exploration-Lab/CS-781-Capstone` (stored as a Kaggle secret named `HF`).
3. Set `RUN_MODE` to `"smoke"`, `"ablation"` or `"full"` according to the desired experiment scale.
4. Execute the notebook top-to-bottom. The data-loading cells download the RR splits for both domains, build the shared label vocabulary, encode every sentence, and construct the document-level tensors.
5. The experiment cells call the unified training routine for each configuration. Each call builds the adapted model, trains (or resumes), evaluates on both domains, writes a submission zip when requested, and stores a detailed result JSON.
6. After all runs finish, the aggregation cells produce a ranked table, an error analysis of the best model, and the final submission artefacts required by Codabench.

All randomness is controlled by a single seed; CUDA determinism flags are set so that repeated runs under identical hardware produce matching trajectories.

