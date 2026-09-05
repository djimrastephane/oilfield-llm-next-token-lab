# Oilfield LLM Next-Token Lab

A hands-on notebook series that shows, with real numbers read directly out of
a real open-source language model, how an LLM predicts its next token and how
it decides which token to actually emit — using realistic drilling,
completions, and well-intervention sentences.

## Purpose

Oil and gas professionals increasingly work alongside LLM-powered tools, but
it's easy to imagine an LLM as a lookup table or a keyword classifier. It
isn't. These notebooks open up a real, locally-run model and show the actual
mechanics: tokenization, logits, softmax, the resulting probability
distribution, and the decoding strategies (greedy decoding, sampling,
temperature, top-k/top-p) that turn that distribution into generated text.

Every probability, logit, and token shown is read directly out of the loaded
model. Nothing is hard-coded, simulated, or adjusted to produce a tidier-
looking result.

## Notebooks in this series

1. **`01_how_a_real_llm_predicts_the_next_token.ipynb`** — the next-token
   prediction step only: prompt → tokenizer → logits → softmax → top
   candidate tokens. Then compares oilfield contexts (e.g. why "POOH" alone
   doesn't tell you what operation is happening) to show how context shifts
   the distribution.
2. **`02_temperature_sampling_and_decoding_strategies.ipynb`** — takes the
   probability distribution from notebook 1 and shows how a token actually
   gets selected from it: greedy decoding vs. sampling, temperature, top-k
   and top-p (nucleus) sampling, and a small, fully transparent manual
   generation loop that produces real multi-token oilfield completions.
3. **`03_embeddings_and_attention.ipynb`** — opens the transformer box a
   little further: real token embedding vectors and their cosine
   similarities (including the honest finding that raw embeddings mostly
   track spelling, not oilfield meaning), and real attention weights
   extracted layer by layer, with an explicit, sustained caution against
   treating attention as a causal explanation of a prediction.
4. **`04_gradient_attribution_and_occlusion.ipynb`** — computes methods that
   are mathematically tied to the model's actual output: gradient × input,
   Integrated Gradients (validated first on a toy function with a known
   exact answer, then checked against its own completeness guarantee on the
   real model — a check it does not cleanly pass, reported honestly rather
   than hidden), and direct occlusion. Compares all three, and states
   plainly what none of them can establish on their own.

Deeper interpretability methods (activation patching, causal tracing,
probing) are left for a possible future notebook so this series stays small,
focused, and fully verifiable.

## Target audience

- Oil and gas engineers, operations staff, and managers with little or no
  programming background who want an honest, concrete picture of what an LLM
  actually does.
- Data scientists who want a technically rigorous but visually simple
  reference example using domain-realistic text.

No machine learning background is assumed. Some comfort reading Python
output (tables, simple charts) is helpful but not required.

## What's in this repo

```
oilfield-llm-next-token-lab/
├── notebooks/
│   ├── 01_how_a_real_llm_predicts_the_next_token.ipynb
│   ├── 02_temperature_sampling_and_decoding_strategies.ipynb
│   ├── 03_embeddings_and_attention.ipynb
│   └── 04_gradient_attribution_and_occlusion.ipynb
├── requirements.txt
└── README.md
```

## Installation

1. Make sure you have Python 3.10+ installed.
2. From the repo root, create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to run

```bash
jupyter notebook notebooks/01_how_a_real_llm_predicts_the_next_token.ipynb
jupyter notebook notebooks/02_temperature_sampling_and_decoding_strategies.ipynb
jupyter notebook notebooks/03_embeddings_and_attention.ipynb
jupyter notebook notebooks/04_gradient_attribution_and_occlusion.ipynb
```

Each notebook runs independently, top to bottom — none require another
notebook in the series to have been run first; each repeats the model-loading
and helper-function setup it needs so it stands on its own. The first code
cell that loads the model will download it from Hugging Face the first time
you run any notebook; after that it's cached and reused. Note that notebook 3
loads the same cached weights differently (`attn_implementation="eager"`,
`float32`) than notebooks 1–2, for reasons explained in its own Section 2 —
no extra download is needed, but it runs somewhat slower as a result.
Notebook 4 is also slower than notebooks 1–2: its Integrated Gradients
sections each run dozens to hundreds of forward-and-backward passes, taking
roughly a minute in total on Apple Silicon (longer on CPU-only machines).

## Model and hardware expectations

- **Model:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
  an open-source instruction-tuned model, run entirely locally via Hugging
  Face Transformers + PyTorch. No external API calls are made for inference.
- **Download size:** approximately 3 GB (downloaded once, cached under
  `~/.cache/huggingface`).
- **Hardware:** runs on a modern MacBook, including Apple Silicon (the
  notebook automatically uses the `mps` backend when available), Intel Macs,
  or any machine with a CUDA GPU. CPU-only execution also works — it's just
  slower.
- **Offline use:** after the first download, the notebook runs fully offline.
- If `Qwen2.5-1.5B-Instruct` fails to load in your environment for any
  reason, the notebook automatically falls back to a smaller sibling model
  and clearly reports which model actually ran — it will never silently
  substitute a model without telling you.

## Learning objectives

After working through **notebook 1** you should be able to explain:

1. What a language model actually predicts (tokens, not ideas).
2. What a token is, and why it isn't the same thing as a word.
3. What a logit is, and how it differs from a probability.
4. How softmax converts logits into a probability distribution.
5. What the real top candidate next tokens look like for an oilfield
   sentence, straight from the model.
6. Why changing the surrounding context (e.g. drilling vs. intervention)
   changes the model's predictions — using "POOH" as the running example.
7. Why this behavior can't be replicated with a simple keyword rule.
8. Why we should avoid claiming to know exactly *why* an internal neural
   network chose a particular token, and what we can and can't legitimately
   infer from observed probability changes.

After working through **notebook 2** you should additionally be able to
explain:

1. What greedy decoding is, and how it relates to `argmax`.
2. What sampling is, and why the highest-probability token isn't guaranteed
   to be picked.
3. What temperature does mathematically to a distribution — and that it
   reshapes the model's existing output rather than changing the model.
4. What top-k and top-p (nucleus) sampling keep and discard, using the
   model's real probabilities.
5. How repeating a single decoding step, one token at a time, is what
   generation actually is.

After working through **notebook 3** you should additionally be able to
explain:

1. What a token embedding vector is, and what similarity between two
   embeddings does (and does not) tell you.
2. Why raw input-embedding nearest-neighbors often reflect spelling and
   sub-word structure rather than oilfield meaning.
3. What attention is, mechanically, inside a transformer layer.
4. What real attention weights look like for an oilfield sentence, and how
   they can differ across layers.
5. Why "the model attended to X, therefore that's why it predicted Y" is an
   overclaim, and what can be said more modestly instead.

After working through **notebook 4** you should additionally be able to
explain:

1. What gradient-based attribution is, and how it's mathematically
   connected to the model's output in a way attention isn't.
2. What "gradient × input" measures, and its key limitation (saturation).
3. What Integrated Gradients' completeness axiom claims, how to check it
   with real numbers, and what it means when a theoretically well-motivated
   method fails that check in practice.
4. What occlusion-based attribution measures, and how it differs in kind
   from gradient-based methods.
5. Why these methods disagreeing with each other — and with attention from
   notebook 3 — is expected, not a contradiction to explain away.
