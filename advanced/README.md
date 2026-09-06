# Advanced path — interpretability research

**This folder is optional.** It's for data scientists and ML practitioners
who want to look inside a real language model's internals — not required
to get practical value from the main, five-notebook series in
`../notebooks/`, which is complete on its own: it runs from "what does an
LLM actually do?" through decoding strategies, why confident answers can
still be wrong, grounding answers in real documents (RAG), and a
practical day-to-day rules summary. See the top-level
[README](../README.md) for that series.

Where the main path answers "what happens, and how do I use it well?", this
path answers "what's actually going on inside the network, and how
rigorously can we claim to know that?" Every notebook here follows the same
rule as the main path — every number comes from the real, loaded model —
but the questions get progressively more technical, and each one assumes
comfort with Python, PyTorch tensors, and general deep-learning concepts
(attention, gradients, embeddings).

**Assumed background:** Python and PyTorch fluency; familiarity with
transformer architecture (attention, embeddings, MLP/feed-forward blocks);
comfort reading logits, gradients, and linear algebra notation.

## Notebooks

1. **`03_embeddings_and_attention.ipynb`** — real token embedding vectors
   and their cosine similarities (a genuinely useful connection to
   semantic search / RAG — see the main path's
   [`04_grounding_answers_in_real_documents.ipynb`](../notebooks/04_grounding_answers_in_real_documents.ipynb)
   for the practical version — plus the honest finding that raw embeddings
   mostly track spelling, not oilfield meaning), and real per-layer
   attention weights, with a sustained caution against treating attention
   as a causal explanation.
2. **`04_gradient_attribution_and_occlusion.ipynb`** — gradient × input,
   Integrated Gradients (validated on a toy function, then checked against
   its own completeness guarantee on the real model — a check it fails,
   reported honestly), and direct occlusion.
3. **`05_activation_patching_and_causal_tracing.ipynb`** — a real
   intervention on the model's running computation: cache a clean run,
   corrupt part of the input, and patch pieces back in to see what's
   causally recoverable where.
4. **`06_probing_classifiers.ipynb`** — trains linear classifiers on the
   model's frozen internal representations to test whether an
   externally-defined category is linearly decodable — and whether that
   holds up under a stricter, harder evaluation split.
5. **`07_individual_head_circuit_analysis.ipynb`** — isolates and ablates
   individual attention heads (336 of them) to find which single heads
   causally matter most, then checks whether that head's own attention
   pattern tells a consistent story.
6. **`08_individual_neuron_analysis.ipynb`** — screens thousands of
   individual MLP neurons cheaply, validates the top candidates with real
   ablation, and interprets the winner two independent ways (causal
   ablation and a prompt-free "logit lens" projection) — which agree.

## How to run

Same environment as the main path — see the top-level
[README](../README.md) for installation. `06_probing_classifiers.ipynb`
additionally needs `scikit-learn` (already listed in `requirements.txt`).
Each notebook runs independently, top to bottom. Notebooks 3 and 7 load the
model differently (`attn_implementation="eager"`, `float32`) than the rest,
for reasons explained in each notebook's own setup section.
