# CS-781 Capstone Phase 2: Legal Named Entity Recognition (LNER)

This notebook implements **Phase 2** of the CS-781 Capstone Project: fine-tuning the Phase-1 pretrained legal language models for **Legal Named Entity Recognition (LNER)**.

The task is token-level sequence labeling (BIO scheme) on long Indian legal documents. A classification head is placed on top of a frozen or partially/fully updated Phase-1 decoder backbone. Multiple parameter-efficient fine-tuning (PEFT) techniques, loss functions, attention masks, context windows, and decoding strategies are systematically compared. The best model on the validation set is used to produce the final Codabench submission.

## Task and Data

Data reside under `Phase_2_LNER` in the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. Each document is a long legal text accompanied by gold character spans of the form `(start, end, label)`. Labels follow a standard BIO scheme; the precise label inventory is derived from the training set at runtime and determines `NUM_LABELS`.

Because documents routinely exceed the Phase-1 context length, every document is split into overlapping windows of length `WINDOW_LEN = 512` tokens (consecutive windows share a 50 % overlap). A leading `<s>` token is prepended to each window. During training the windows are treated as independent examples; at inference the per-token log-probabilities from all overlapping windows that cover a given position are averaged before decoding.

Three run modes control data volume and training length:
- `full` – complete data and the intended number of epochs (production runs and submissions);
- `ablation` – seeded 30 % / 50 % slices of train/validation with a hard epoch cap (fast method comparison);
- `smoke` – a handful of documents and a single epoch (end-to-end pipeline verification).

## Phase-1 Backbone Architectures

Four pretrained checkpoints are available. Their configurations are stored in `PHASE1_CONFIGS` and loaded by name:

- `gqa-rope` – 8 layers, standard GELU FFN, grouped-query attention + RoPE;
- `mla-sin` – 12 layers, multi-head latent attention, sinusoidal positional encodings;
- `gqa-rope-swiglu-moe` – 12 layers, SwiGLU + Mixture-of-Experts (4 experts, top-2), GQA + RoPE (the default backbone for technique experiments);
- `swiglu-moe-rope-conv` – same as above plus a parallel depth-wise causal convolution (kernel 25) inside each block.

All backbones share `d_model = 768`, 12 query heads (4 KV heads for GQA variants), vocabulary size 20 000 and a maximum sequence length of 2048. The Phase-1 `DecoderLM` class (including RoPE, MLA, MoE, and the optional local-convolution pathway) is reused unchanged; only the language-model head is replaced by a token-classification head.

## Adaptation Methods

A single `Experiment` dataclass completely specifies every training run. The field `method` selects one of the following adaptation strategies applied to a frozen or partially-frozen Phase-1 backbone:

- **full** – every parameter is trainable (performance ceiling and knowledge-distillation teacher);
- **partial** – only the top `n_top` blocks plus the final layer-norm and the new classification head are updated;
- **unfreeze** – gradual unfreezing: an increasing number of blocks from the top become trainable across epochs;
- **bitfit** – only bias terms (and the classification head) are trained; requires a backbone that still contains biases;
- **lora** – low-rank adapters (`rank`, `alpha`, dropout) inserted into the linear projections of every attention and feed-forward module;
- **lorma** – a LoRA variant that also adapts the RMSNorm / LayerNorm scale parameters;
- **adalora** – adaptive-rank LoRA that begins with a full rank budget and prunes to a target ratio (`adalora_final_ratio`) according to an importance score, with an orthogonality regulariser on the low-rank factors;
- **adapter** – bottleneck adapters of hidden dimension `adapter_dim` inserted after each block.

An optional `block_stride` (default 1) keeps only every *k*-th block, producing a thinner student that can be distilled from a full teacher. Knowledge distillation itself is activated by supplying a `teacher_name`; the student then optimises a mixture of hard-label cross-entropy and temperature-scaled KL divergence on the soft teacher logits.

The attention mask inside the backbone can be left causal (the original left-to-right pretraining mask) or made bidirectional by constructing a key-padding mask from the attention mask. Both settings are ablated.

## Token Classification Head and Loss

A lightweight `NERHead` sits on top of the adapted backbone. It applies dropout followed by a linear projection from `d_model` onto the label vocabulary. The head is always trained; the backbone parameters that receive gradients depend on the chosen method.

Class imbalance is severe (the majority of tokens are outside any entity). Three loss variants are supported:

- plain cross-entropy;
- inverse-frequency weighted cross-entropy (weights raised to a power `cw_power` and clipped at `CW_CAP`);
- focal loss that further down-weights easy examples while still using the class weights.

Ignored positions (padding, the leading `<s>`, and pure-whitespace tokens) receive the label `-100` and are excluded from the loss.

## Decoding

After the overlapping windows have been averaged, two decoding strategies are available:

- greedy argmax per token;
- Viterbi decoding that respects the BIO transition constraints (an I-X tag may follow only a B-X or I-X of the same type).

The chosen decoder produces a sequence of BIO tags that is converted back into character spans using the original token-to-character offset mapping. Strict span-level precision, recall and macro-F1 (exact boundary and type match) are the primary metrics; a secondary greedy-versus-Viterbi comparison is recorded for every run.

## Training Loop and Checkpointing

Windows are collated into batches of size `BATCH_SIZE` (default 8). The optimiser is AdamW with separate learning rates for backbone/adapter parameters and for the classification head, weight decay only on the former, a linear warm-up followed by cosine decay, and gradient clipping. Mixed-precision training is optional; inference always uses FP16 autocast when a GPU is present.

Every epoch the model is evaluated on the validation set with strict span F1. The best checkpoint (by validation F1) and a full resume checkpoint (model, optimiser, scaler, AdaLoRA importance scores, RNG state) are written atomically. Training can therefore be interrupted and resumed without loss of reproducibility. After the final epoch the best weights are restored before test-set inference.

## Experimental Protocol

A typical campaign proceeds as follows.

1. Full bidirectional fine-tuning of the main backbone (`gqa-rope-swiglu-moe`) yields both the strongest single model and the teacher for distillation.
2. A thinner student (every second block) is trained with and without distillation from that teacher.
3. The suite of parameter-efficient methods (partial, LoRA, LoRMA, adapter, BitFit, AdaLoRA, gradual unfreezing) is run on the same backbone.
4. The strongest PEFT method (LoRMA) is repeated on every Phase-1 architecture to measure backbone sensitivity.
5. Controlled ablations examine causal versus bidirectional attention, window sizes 256 / 512 / 1024, and the three loss formulations.

All runs share the same random seed, the same data splits (under a given `RUN_MODE`), and the same evaluation code, so differences are attributable only to the experimental factor under study. Results are serialised to JSON; a helper aggregates them into a pandas table sorted by validation F1 and writes the single best prediction file together with one “best-per-architecture” submission zip.

## Running the Code

1. Place the Phase-1 tokenizer (`tokenizerv3.pkl`) and the four model checkpoints into the working directory (or adjust the copy commands at the top of the notebook).
2. Authenticate with a Hugging Face token that can read `Exploration-Lab/CS-781-Capstone` (stored as a Kaggle secret named `HF`).
3. Set `RUN_MODE` to `"smoke"`, `"ablation"` or `"full"` according to the desired experiment scale.
4. Execute the notebook top-to-bottom. The data-loading cells download the LNER splits, build the BIO label vocabulary, encode every document into token IDs and character offsets, and construct the sliding windows.
5. The experiment cells call `run_experiment` for each configuration. Each call builds the adapted model, trains (or resumes), evaluates, writes a submission zip when requested, and stores a detailed result JSON.
6. After all runs finish, the aggregation cells produce a ranked table, an error breakdown of the best model, and the final `submission.zip` required by Codabench.

All randomness is controlled by a single seed; CUDA determinism flags are set so that repeated runs under identical hardware produce matching trajectories.

