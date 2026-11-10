# CS-781 Capstone Phase 5: Legal Statute Identification (LSI)

This notebook implements **Phase 5** of the CS-781 Capstone Project: fine-tuning the Phase-1 pretrained legal language models for **Legal Statute Identification (LSI)**.

The task is **multi-label classification**: given a long court case, predict the set of relevant statutes (from a fixed inventory of statute identifiers). Because cases are extremely long, a hierarchical architecture is employed—Phase-1 decoder chunks are encoded independently and then pooled into a single case-level representation that feeds a multi-label classification head.

Multiple parameter-efficient fine-tuning (PEFT) techniques, chunk-selection strategies, pooling / head variants, loss functions, sampling strategies, calibration methods, and knowledge-distillation setups are systematically compared. Global and per-label decision thresholds are tuned on the development set. The best models produce the final Codabench submissions.

## Task and Data

Data reside under `Phase_5_LSI` in the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. Each example is identified by an `id`, contains the full case text, and carries a list of applicable statute identifiers under the key `labels`. The complete inventory of statutes is derived from the training set; statutes that never appear (or appear fewer than a minimum number of times) are marked non-predictable and are excluded from evaluation and from the final submission.

Because case texts are long, every document is segmented into overlapping chunks of 512 tokens (including a leading `<s>`). Consecutive chunks share an overlap of 64 tokens. At most `DEFAULT_MAX_CHUNKS = 8` chunks are retained per case; the selection policy (head, tail, head_tail or uniform) is an experimental factor. A smoke-test flag restricts the data to tiny subsets so that the whole pipeline can be verified in minutes.

## Phase-1 Backbone Architectures

Four pretrained checkpoints are available and are loaded by name from `PHASE1_CONFIGS`:

- `gqa-rope` – 8 layers, standard GELU FFN, grouped-query attention + RoPE;
- `mla-sin` – 12 layers, multi-head latent attention, sinusoidal positional encodings;
- `gqa-rope-swiglu-moe` – 12 layers, SwiGLU + Mixture-of-Experts (4 experts, top-2), GQA + RoPE (the default backbone for technique experiments);
- `swiglu-moe-rope-conv` – same as the MoE model plus a parallel depth-wise causal convolution (kernel 25) inside each block.

All backbones share `d_model = 768`, 12 query heads (4 KV heads for GQA variants) and a maximum sequence length of 2048. The Phase-1 `DecoderLM` implementation is reused unchanged; only the language-model head is replaced by a multi-label statute classifier.

## Long-Document Representation

Each retained chunk is encoded independently by the Phase-1 backbone. The resulting token hidden states are aggregated into a chunk embedding by mean pooling (or by a learned attention pool). The set of chunk embeddings that belong to one case is then combined by one of several head types:

- simple mean across chunks;
- a learned attention pool;
- a bidirectional GRU that reads the sequence of chunk vectors and returns its final state.

The aggregated case vector is projected by a linear layer onto the full statute vocabulary, producing a logit for every statute. Because the number of chunks is small, the hierarchical stage adds negligible cost while allowing the model to weigh different sections of the judgment.

## Adaptation Methods

A single `Experiment` dataclass completely specifies every run. The field `method` selects one of the following strategies applied to the Phase-1 backbone:

- full fine-tuning of every parameter;
- partial unfreezing of the top blocks;
- LoRA and LoRMA (low-rank adapters; LoRMA also adapts LayerNorm / RMSNorm scales);
- AdaLoRA (adaptive-rank LoRA with importance-based pruning);
- bottleneck adapters;
- prompt / prefix tuning (soft tokens prepended to every chunk);
- QLoRA-style quantised low-rank adaptation when memory is constrained.

Knowledge distillation is supported by first exporting the teacher’s per-statute logits on the training set and then training a thinner student (every second block retained) against a mixture of hard labels and temperature-scaled soft targets. The same chunk-selection strategy and maximum chunk count are enforced for teacher and student so that the distilled signal remains aligned.

## Loss Functions and Class Imbalance

Statute frequencies are extremely long-tailed. Three loss variants are implemented:

- ordinary binary cross-entropy;
- inverse-frequency weighted binary cross-entropy (positive weights clipped at `POS_WEIGHT_CAP`);
- focal loss that further down-weights easy negatives.

In addition, a weighted random sampler can oversample cases that contain rare statutes (those appearing in fewer than 2 % of the training set). The combination of loss weighting and rare-statute oversampling is the default for production runs.

## Calibration and Thresholding

After training, a temperature is fitted on the development logits to improve probability calibration. Two threshold regimes are then tuned:

- a single global threshold that maximises macro-F1 across all statutes;
- a per-statute threshold (only for labels that have at least `PER_LABEL_MIN_SUPPORT` positive development examples).

The submission mode (`THRESHOLD_MODE`) selects which set of thresholds is applied to the test probabilities. All logits, temperatures and thresholds are serialised so that alternative decision rules or ensembles can be constructed later without retraining.

## Training Loop and Checkpointing

Cases are processed in micro-batches of `BATCH_DOCS` documents; gradient accumulation yields the effective batch size. The optimiser is AdamW with separate learning rates for backbone/adapter parameters and for the classification head, a short linear warm-up followed by cosine decay, and gradient clipping. Gradient checkpointing is enabled by default to keep memory consumption manageable on long multi-chunk inputs. Every epoch the model is evaluated on the development set; the best checkpoint and a full resume checkpoint are written atomically.

## Experimental Protocol

A typical campaign proceeds as follows.

1. Full fine-tuning of the main backbone produces both the strongest single model and the teacher whose logits are exported for distillation.
2. A half-depth student is trained with and without distillation from that teacher.
3. The suite of parameter-efficient methods (LoRA, LoRMA, AdaLoRA, adapters, prompt/prefix tuning, partial unfreezing) is run on the same backbone.
4. The strongest PEFT method is repeated on every Phase-1 architecture.
5. Controlled ablations examine chunk-selection strategy, hierarchical head type, loss formulation and rare-statute oversampling.

All runs share the same random seed and the same data splits, so observed differences are attributable only to the experimental factor under study. Results are serialised to JSON; helper routines aggregate them into a ranked table, produce section-probe diagnostics (macro-F1 when only head, tail, middle, \ldots chunks are shown), and write the required Codabench submission zips.

## Running the Code

1. Place the Phase-1 tokenizer (`tokenizerv3.pkl`) and the four model checkpoints into the working directory (or adjust the copy commands at the top of the notebook).
2. Authenticate with a Hugging Face token that can read `Exploration-Lab/CS-781-Capstone`.
3. Set `SMOKE_TEST = True` for a rapid end-to-end check or `False` for full-scale experiments.
4. Execute the notebook top-to-bottom. The data-loading cells download the LSI splits, build the statute vocabulary, encode every case into overlapping chunks, and construct the multi-label target matrices.
5. The experiment cells call the unified training routine for each configuration. Each call builds the adapted model, trains (or resumes), calibrates, tunes thresholds, writes a submission zip, and stores logits, calibration parameters and a detailed result JSON.
6. After all runs finish, the aggregation cells produce a ranked table, section-probe analyses and the final submission artefacts.

All randomness is controlled by a single seed; CUDA determinism flags are set so that repeated runs under identical hardware produce matching trajectories.