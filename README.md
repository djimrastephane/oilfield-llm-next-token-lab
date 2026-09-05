# Oilfield LLM Next-Token Lab

A hands-on notebook that shows, with real numbers from a real open-source
language model, how an LLM predicts its next token — using realistic
drilling, completions, and well-intervention sentences.

## Purpose

Oil and gas professionals increasingly work alongside LLM-powered tools, but
it's easy to imagine an LLM as a lookup table or a keyword classifier. It
isn't. This notebook opens up a real, locally-run model and shows the actual
mechanics of a single next-token prediction: tokenization, logits, softmax,
and the resulting probability distribution — and then shows how that
distribution shifts when the surrounding oilfield context changes (e.g. why
"POOH" alone doesn't tell you what operation is happening).

Every probability, logit, and token shown in the notebook is read directly
out of the loaded model. Nothing is hard-coded, simulated, or adjusted to
produce a tidier-looking result.

**Scope note:** this is notebook 1 of a planned series. It deliberately
covers *only* the next-token prediction step (prompt → tokenizer → logits →
softmax → top tokens → context comparison). Decoding strategies (greedy vs.
sampling, temperature, top-k/top-p) and deeper interpretability methods
(attention, embeddings, attribution) are left for follow-up notebooks so this
one stays small, focused, and fully verifiable.

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
│   └── 01_how_a_real_llm_predicts_the_next_token.ipynb
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
```

Run the cells in order, top to bottom. The first code cell that loads the
model will download it from Hugging Face the first time you run it.

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

After working through the notebook you should be able to explain:

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
