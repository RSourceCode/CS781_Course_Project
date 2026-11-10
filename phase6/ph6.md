# CS-781 Capstone Phase 6: Legal Document Summarization

This notebook implements **Phase 6** of the CS-781 Capstone Project: fine-tuning the Phase-1 pretrained legal language models for **abstractive summarization of long Indian court judgments**.

The model is trained in a standard encoder-decoder-style causal language-modeling setup: the input sequence is formed as

```
<s>  [document tokens]  [SEP]  [summary tokens]  </s>
```

all inside the 2 048-token context window of the Phase-1 backbone. Because source documents are far longer than this budget, several long-document packing strategies (head/tail, head-tail, uniform windows, sentence selection, and a hierarchical “summary-of-summaries” pipeline) are explored. Multiple parameter-efficient fine-tuning (PEFT) techniques and knowledge distillation are systematically compared. Generated summaries are evaluated with ROUGE, BERTScore and negative log-likelihood; the best models produce the final Codabench submissions.

## Task and Data

Data reside under `Phase_6_Summ` in the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. Each example is identified by an `id`, contains the full judgment under the key `document`, and carries a reference summary under the key `summary`. Reference summaries are truncated to a maximum of 256 tokens; generation is likewise capped at this length.

Because judgments routinely exceed the Phase-1 context window of 2048 tokens, several long-document input strategies are supported. The training sequence is always formed as  
`<s> [selected document tokens] [SEP] [summary tokens] </s>`  
and is constrained to fit inside `TRAIN_BLOCK_SIZE`. A held-out validation split is carved from the official training file (10 % of the data, bounded between 50 and 300 examples). Only a subset of the validation documents (`N_VAL_GEN = 100`) is actually decoded and scored with ROUGE and BERTScore, keeping evaluation tractable.

A smoke-test flag restricts the data to tiny subsets so that the whole pipeline can be verified in minutes.

## Phase-1 Backbone Architectures

Four pretrained checkpoints are available and are loaded by name from `PHASE1_CONFIGS`:

- `gqa-rope` – 8 layers, standard GELU FFN, grouped-query attention + RoPE;
- `mla-sin` – 12 layers, multi-head latent attention, sinusoidal positional encodings;
- `gqa-rope-swiglu-moe` – 12 layers, SwiGLU + Mixture-of-Experts (4 experts, top-2), GQA + RoPE (the default backbone for technique experiments);
- `swiglu-moe-rope-conv` – same as the MoE model plus a parallel depth-wise causal convolution (kernel 25) inside each block.

All backbones share `d_model = 768`, 12 query heads (4 KV heads for GQA variants) and a maximum sequence length of 2048. The Phase-1 `DecoderLM` implementation is reused unchanged; the language-model head continues to predict the next token, now conditioned on a document prefix.

## Long-Document Input Strategies

Because a full judgment cannot fit inside the context window, the document portion of the training sequence is constructed by one of the following policies:

- head – take tokens from the beginning of the judgment;
- tail – take tokens from the end;
- head_tail – allocate a configurable fraction of the budget to the beginning and the remainder to the end;
- uniform – sample a fixed number of evenly spaced windows and concatenate them;
- sentence_select – always keep a fraction of the budget from the start and from the end, filling the middle with the highest-scoring sentences according to a simple salience heuristic.

At inference time an optional hierarchical procedure is also available: the document is first segmented into non-overlapping blocks of 1024 tokens, each block is summarised to at most 96 tokens, and the resulting segment summaries are themselves summarised to produce the final output. This two-stage process allows the model to cover judgments that are many times longer than the context window.

## Adaptation Methods

A single `Experiment` dataclass completely specifies every run. The field `method` selects one of the following strategies applied to the Phase-1 backbone:

- full fine-tuning of every parameter (quality ceiling and knowledge-distillation teacher);
- LoRA and AdaLoRA (low-rank adapters; AdaLoRA additionally prunes ranks according to an importance score);
- Houlsby and Pfeiffer bottleneck adapters;
- prompt tuning and prefix tuning (soft tokens prepended to every sequence);
- QLoRA-style quantised low-rank adaptation when memory is constrained.

Knowledge distillation is supported by training a thinner student (every second block retained) against a mixture of hard next-token labels and temperature-scaled soft targets obtained from a fully fine-tuned teacher. The same document-selection strategy is used for teacher and student so that the distilled signal remains aligned.

## Training Objective and Generation

Training minimises the standard autoregressive negative log-likelihood of the reference summary tokens, conditioned on the selected document prefix. Gradient checkpointing is enabled by default. The optimiser is AdamW with separate learning rates for backbone/adapter parameters and for any newly introduced soft-prompt parameters, a short linear warm-up followed by cosine decay, and gradient clipping. Mixed precision keeps memory consumption manageable.

At inference the model generates summaries by autoregressive decoding (temperature sampling or greedy decoding) up to the 256-token limit. Special tokens are banned from being sampled. Generated summaries are evaluated with ROUGE-1, ROUGE-L and BERTScore (F1) against the reference; average summary length is also recorded.

## Experimental Protocol

A typical campaign proceeds as follows.

1. Full fine-tuning of the main backbone produces both the strongest single model and the teacher for distillation.
2. A half-depth student is trained with and without distillation from that teacher.
3. The suite of parameter-efficient methods (LoRA, AdaLoRA, Houlsby/Pfeiffer adapters, prompt/prefix tuning) is run on the same backbone and, for the strongest methods, on every Phase-1 architecture.
4. Controlled ablations examine the effect of the long-document selection strategy and of hierarchical versus single-pass decoding.

All runs share the same random seed and the same validation split, so observed differences are attributable only to the experimental factor under study. Results are serialised to JSON; helper routines aggregate them into a ranked table and write the required Codabench submission zips.

## Running the Code

1. Place the Phase-1 tokenizer (`tokenizerv3.pkl`) and the four model checkpoints into the working directory (or adjust the copy commands at the top of the notebook).
2. Authenticate with a Hugging Face token that can read `Exploration-Lab/CS-781-Capstone`.
3. Set `SMOKE_TEST = True` for a rapid end-to-end check or `False` for full-scale experiments.
4. Execute the notebook top-to-bottom. The data-loading cells download the summarization splits, build the validation partition, encode documents and summaries, and construct the concatenated training sequences according to the chosen selection strategy.
5. The experiment cells call the unified training routine for each configuration. Each call builds the adapted model, trains (or resumes), generates validation summaries, computes ROUGE and BERTScore, writes a submission zip, and stores a detailed result JSON.
6. After all runs finish, the aggregation cells produce a ranked table and the final submission artefacts.

All randomness is controlled by a single seed; CUDA determinism flags are set so that repeated runs under identical hardware produce matching trajectories.
