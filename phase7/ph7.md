# CS-781 Capstone Phase 7: Precedent Case Retrieval (PCR)

This notebook implements **Phase 7** of the CS-781 Capstone Project: fine-tuning the Phase-1 pretrained legal language models for **Precedent Case Retrieval (PCR)**.

Given a query judgment, the system must retrieve the most relevant prior cases from a candidate pool. The task is formulated as **dense retrieval**: both queries and candidate documents are encoded into fixed-dimensional vectors (or late-interaction representations) by a Phase-1 backbone; candidates are ranked by cosine similarity or MaxSim. The top-7 ranked candidates are submitted for each query.

Multiple parameter-efficient fine-tuning (PEFT) techniques, pooling strategies, similarity functions, contrastive losses, hard-negative mining schemes, and knowledge-distillation variants are systematically compared. The best models produce the final Codabench submissions.


## Task and Data

Data reside under `Phase_7_PCR` in the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. Each split supplies a collection of query cases and a pool of candidate documents. Every query is accompanied by a list of relevant candidate identifiers. At test time the model ranks the entire pool for each query and the top seven identifiers are submitted.

Because both queries and candidates are long legal documents, every text is segmented into overlapping chunks of 512 tokens (including the leading `<s>` and trailing `</s>`). Trailing fragments shorter than 32 tokens are discarded. At most `DEFAULT_MAX_CHUNKS = 4` chunks are retained per document; the selection policy (head, tail, head_tail or uniform) is an experimental factor. A smoke-test flag restricts the data to tiny subsets so that the whole pipeline can be verified in minutes.

## Phase-1 Backbone Architectures

Four pretrained checkpoints are available and are loaded by name from `PHASE1_CONFIGS`:

- `gqa-rope` – 8 layers, standard GELU FFN, grouped-query attention + RoPE;
- `mla-sin` – 12 layers, multi-head latent attention, sinusoidal positional encodings;
- `gqa-rope-swiglu-moe` – 12 layers, SwiGLU + Mixture-of-Experts (4 experts, top-2), GQA + RoPE (the default backbone for technique experiments);
- `swiglu-moe-rope-conv` – same as the MoE model plus a parallel depth-wise causal convolution (kernel 25) inside each block.

All backbones share `d_model = 768`, 12 query heads (4 KV heads for GQA variants) and a maximum sequence length of 2048. The Phase-1 `DecoderLM` implementation is reused unchanged; the language-model head is replaced by a retrieval head that produces dense representations for cosine or MaxSim scoring.

## Document Encoding and Similarity

Each retained chunk is passed through the backbone. Token hidden states inside a chunk are aggregated by mean pooling or by taking the final token representation. The resulting chunk vectors of one document are then combined by mean pooling or by a learned attention pool to obtain a single document embedding. Two similarity functions are supported:

- cosine similarity between the pooled query and document vectors;
- MaxSim (ColBERT-style late interaction) that computes the sum of maximum token-level similarities between the query and every document token.

The cosine pathway is cheaper at indexing time and is the default; MaxSim is retained as an ablation that can capture finer-grained matches.

## Training Objective and Hard Negatives

Training uses an in-batch contrastive loss. For every query the positive is its labelled relevant candidate; additional hard negatives are supplied by one of three strategies:

- none – only the in-batch documents serve as negatives;
- random – random non-relevant documents drawn from the pool;
- tfidf – TF-IDF retrieval is run once offline; for each query the ranks lying between a skip offset and a pool size among the non-relevant top candidates become the hard-negative pool.

Both InfoNCE (with a temperature hyper-parameter) and a classic triplet loss (with a margin) are implemented. Knowledge distillation from a fully fine-tuned teacher is also supported: the student matches both the teacher’s embedding space and the soft similarity distribution the teacher produces over the training pool.

## Adaptation Methods

A single `Experiment` dataclass completely specifies every run. The field `method` selects one of the following strategies applied to the Phase-1 backbone:

- full fine-tuning of every parameter (quality ceiling and distillation teacher);
- partial unfreezing of the top blocks;
- LoRA, LoRMA and QLoRA (low-rank adapters);
- IA³ (learned scaling vectors);
- prompt tuning and prefix tuning (soft tokens prepended to every chunk);
- frozen backbone (zero-shot baseline that uses only the pretrained representations).

The same chunk-selection strategy and maximum chunk count are enforced for teacher and student when distillation is active.

## Evaluation and Submission

After training the model encodes the entire development (or test) pool once. Each query is then scored against every pool document; the ranked list is truncated at the required cut-offs. Metrics reported are F1@7, P@7, R@7, R@20 and R@50. The top-seven identifiers for every test query are written to the Codabench submission format. Encoding latency, scoring latency and the memory footprint of the dense index are also recorded.

## Experimental Protocol

A typical campaign proceeds as follows.

1. Full fine-tuning of the main backbone produces both the strongest single model and the teacher whose embeddings are later distilled.
2. A half-depth student is trained with and without distillation from that teacher.
3. The suite of parameter-efficient methods is run on the same backbone and, for the strongest methods, on every Phase-1 architecture.
4. Controlled ablations examine similarity function (cosine versus MaxSim), token- and chunk-pooling choices, loss formulation, hard-negative strategy and long-document selection policy.
5. Zero-shot baselines that freeze the Phase-1 weights are measured for every architecture to quantify the contribution of continued training.

All runs share the same random seed and the same data splits, so observed differences are attributable only to the experimental factor under study. Results are serialised to JSON; helper routines aggregate them into a ranked table and write the required Codabench submission zips.

## Running the Code

1. Place the Phase-1 tokenizer (`tokenizerv3.pkl`) and the four model checkpoints into the working directory (or adjust the copy commands at the top of the notebook).
2. Authenticate with a Hugging Face token that can read `Exploration-Lab/CS-781-Capstone`.
3. Set `SMOKE_TEST = True` for a rapid end-to-end check or `False` for full-scale experiments.
4. Execute the notebook top-to-bottom. The data-loading cells download the PCR splits, encode every document into overlapping chunks, build the TF-IDF hard-negative index, and construct the training triples.
5. The experiment cells call the unified training routine for each configuration. Each call builds the adapted model, trains (or resumes), evaluates on the development set, ranks the test pool, writes a submission zip, and stores scores together with a detailed result JSON.
6. After all runs finish, the aggregation cells produce a ranked table and the final submission artefacts.

All randomness is controlled by a single seed; CUDA determinism flags are set so that repeated runs under identical hardware produce matching trajectories.
