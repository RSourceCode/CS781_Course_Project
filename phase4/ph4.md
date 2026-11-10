# CS-781 Capstone Phase 4: Court Judgment Prediction and Explanation (CJPE)

This notebook implements **Phase 4** of the CS-781 Capstone Project: fine-tuning the Phase-1 pretrained legal language models for **Court Judgment Prediction / Explanation (CJPE)**.

The task is **binary classification** of long Indian court cases (predicting the judgment outcome). Because individual cases routinely exceed several thousand tokens, a hierarchical long-document architecture is used: the Phase-1 decoder encodes overlapping chunks; a pooling layer then aggregates the chunk representations into a single case-level embedding that is fed to a classification head.

Multiple parameter-efficient fine-tuning (PEFT) techniques, chunk-selection strategies, pooling methods, loss functions, and knowledge-distillation variants are systematically compared. Decision thresholds are tuned on the development set. The best models (and an ensemble) produce the final Codabench submissions.

## Task and Data

Data reside under `Phase_4_CJPE` in the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. Each example is a complete case record identified by an `id`, containing the full judgment text and a binary `label`. Because judgments routinely exceed the Phase-1 context length of 2048 tokens, every document is split into overlapping chunks of 512 tokens (including a leading `<s>`). Consecutive chunks share an overlap of 64 tokens, yielding a stride of 447 content tokens. At most `MAX_CHUNKS = 8` chunks are retained per case; the selection policy is controlled by `CHUNK_STRATEGY` (head, tail, head_tail or uniform sampling).

A smoke-test flag (`SMOKE_TEST`) restricts the data to tiny subsets and writes all artefacts to a separate output directory, allowing the entire pipeline to be verified in minutes.

## Phase-1 Backbone Architectures

Four pretrained checkpoints are available and are loaded by name from `PHASE1_CONFIGS`:

- `gqa-rope` – 8 layers, standard GELU FFN, grouped-query attention + RoPE;
- `mla-sin` – 12 layers, multi-head latent attention, sinusoidal positional encodings;
- `gqa-rope-swiglu-moe` – 12 layers, SwiGLU + Mixture-of-Experts (4 experts, top-2), GQA + RoPE (default backbone for technique experiments);
- `swiglu-moe-rope-conv` – same as the MoE model plus a parallel depth-wise causal convolution (kernel 25) inside each block.

All backbones share `d_model = 768`, 12 query heads (4 KV heads for GQA variants) and a maximum sequence length of 2048. The Phase-1 `DecoderLM` implementation is reused unchanged; only the language-model head is replaced by a case-level classification head.

## Long-Document Representation

Each retained chunk is passed independently through the Phase-1 backbone. The resulting sequence of hidden states is mean-pooled (or attention-pooled) to obtain a fixed-size chunk embedding. The set of chunk embeddings that belong to one case is then aggregated by one of three pooling strategies:

- simple mean pooling across chunks;
- a learned attention pool that produces a weighted sum of chunk vectors;
- a bidirectional GRU that reads the sequence of chunk embeddings and returns its final hidden state (hierarchical modelling).

The aggregated case vector is finally projected by a linear classifier onto the two outcome classes. Dropout is applied before the classifier. Because the number of chunks is small (≤ 8), the hierarchical stage adds negligible compute while allowing the model to weigh different sections of the judgment.

## Adaptation Methods

The same family of parameter-efficient and full-update methods developed in earlier phases is available through a single `Experiment` dataclass:

- full fine-tuning of every parameter;
- partial unfreezing of the top blocks;
- gradual unfreezing across epochs;
- BitFit (bias-only updates);
- LoRA and LoRMA (low-rank adapters, the latter also adapting LayerNorm / RMSNorm scales);
- AdaLoRA (adaptive-rank LoRA with importance-based pruning and an orthogonality regulariser);
- bottleneck adapters.

Knowledge distillation from a fully fine-tuned teacher onto a thinner student (every second block retained) is supported by supplying a `teacher_name`. The attention mask inside the backbone can be left causal or made bidirectional.

## Loss Functions and Decision Threshold

Class imbalance is handled by three loss variants: ordinary cross-entropy, inverse-frequency weighted cross-entropy, and focal loss (`FOCAL_GAMMA = 2.0`). After training, a decision threshold is tuned on the development set to maximise macro-F1; the tuned threshold (rather than the default 0.5) is used for the final test predictions when `USE_TUNED_THRESHOLD` is true.

## Training Loop and Checkpointing

Cases are processed in micro-batches of `BATCH_DOCS` documents. Gradient accumulation over `GRAD_ACCUM` steps yields the effective batch size. The optimiser is AdamW with separate learning rates for backbone/adapter parameters and for the classification head, a short linear warm-up followed by cosine decay, and gradient clipping. Every epoch the model is evaluated on the development set. The best checkpoint (by macro-F1) and a full resume checkpoint are written atomically, so training can be interrupted and resumed without loss of reproducibility.

## Experimental Protocol

A typical campaign proceeds as follows.

1. Full fine-tuning of the main backbone produces both the strongest single model and the teacher for distillation.
2. A half-depth student is trained with and without distillation from that teacher.
3. The suite of parameter-efficient methods is run on the same backbone.
4. The strongest PEFT method is repeated on every Phase-1 architecture.
5. Controlled ablations examine chunk-selection strategy (head / tail / head_tail / uniform), chunk-pooling method (mean / attention / BiGRU), and loss formulation.

All runs share the same random seed and the same data splits, so observed differences are attributable only to the experimental factor under study. Results are serialised to JSON; helper routines aggregate them into a ranked table, perform error analysis by case length and by true/false positive groups, and write Codabench submission zips. An optional ensemble that averages the probability outputs of the top models is also produced.

## Running the Code

1. Place the Phase-1 tokenizer (`tokenizerv3.pkl`) and the four model checkpoints into the working directory (or adjust the copy commands at the top of the notebook).
2. Authenticate with a Hugging Face token that can read `Exploration-Lab/CS-781-Capstone`.
3. Set `SMOKE_TEST = True` for a rapid end-to-end check or `False` for full-scale experiments.
4. Execute the notebook top-to-bottom. The data-loading cells download the CJPE splits, encode every judgment into overlapping chunks, and construct the case-level tensors.
5. The experiment cells call the unified training routine for each configuration. Each call builds the adapted model, trains (or resumes), evaluates, writes a submission zip, and stores a detailed result JSON together with development and test probability arrays.
6. After all runs finish, the aggregation cells produce a ranked table, an error analysis of the best model, section-probe diagnostics, and the final ensemble submission.

All randomness is controlled by a single seed; CUDA determinism flags are set so that repeated runs under identical hardware produce matching trajectories.

