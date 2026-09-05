# Oilfield LLM Next-Token Lab

You type an oilfield sentence into an AI assistant:

> "The crew pulled out of hole with the worn PDC bit and prepared to run a new..."

How does it decide what comes next? Does it look something up in a
drilling manual? Does it follow a rule about "POOH"? Does it "know" the
bit is worn? And why might it give you a different answer if you run the
exact same prompt again tomorrow?

This project lets you look inside a real, locally-run language model and
see what actually happens — using real drilling, completions, and
well-intervention examples, with every number traceable back to the model
itself. Nothing here is invented, simulated, or cleaned up to make the
results look tidier than they are.

Here's the very first real result you'll see, straight from the model,
unfiltered:

```
Input:
"The crew pulled out of hole with the worn PDC bit
 and prepared to run a new ___"

Real next-token candidates (from the actual model):

 bit      ████████████████████████████████████  33.5%
 one      ███████████████████████████████       27.8%
 P        ████████                                7.5%
 (space)  ███                                     2.5%
 drill    ██                                      2.0%
```

Notice "one" is almost as likely as "bit" — a real, slightly surprising
result, not something anyone chose to show you. **These numbers are
probabilities.** To see where they come from, and why they shift when you
change the sentence, keep reading.

## Two paths through this project

| | Main path | Advanced path |
|---|---|---|
| **For** | Oil & gas engineers, operations staff, managers — no programming or ML background needed | Data scientists / ML practitioners who want to look inside the network's internals |
| **Where** | `notebooks/` (2 notebooks) | `advanced/` (6 notebooks) |
| **Time** | ~30–45 minutes | Several hours |
| **Answers** | "What does the model actually do, and how do I use it well?" | "What's mechanistically happening inside, and how rigorously can we claim to know that?" |

**If you're an oil & gas professional, you only need the main path.** The
advanced path is real, technically rigorous interpretability research
(attention, gradient attribution, activation patching, probing
classifiers, individual attention heads and neurons) — genuinely
interesting, but not required to get practical value from this project.
See `advanced/README.md` for that path; the rest of this README is about
the main path.

## The main path: an oilfield professional's journey

```
1. What does an LLM actually do?
        real oilfield sentence -> real model -> real next-token candidates
   |
2. Why does context matter?
        "POOH" alone is ambiguous -- drilling vs. completions vs. intervention
   |
3. Why can the same question get a different answer?
        greedy decoding vs. sampling, temperature, top-k / top-p
   |
4. Why can an LLM sound confident and still be wrong?          } planned --
5. How do we make it safer for engineering use (grounding, RAG)? } not yet
6. What should an engineer remember, day to day?                } built
```

Steps 1–3 are built today, as `notebooks/01_...` and
`notebooks/02_...ipynb`. Steps 4–6 (probability vs. factual correctness,
grounding answers in real engineering documents, and a practical rules
summary) are a planned future addition to this project — not yet built —
listed here so the intended shape of the full journey is clear.

### Notebook 1: What does an LLM actually do, and why does context matter?

`notebooks/01_how_a_real_llm_predicts_the_next_token.ipynb`

Walks through the real mechanics behind the teaser result above: your
sentence gets broken into tokens, the model scores every possible next
token, and those scores turn into the probabilities you saw. Then it uses
oilfield's favorite ambiguous abbreviation — **POOH** ("pull out of
hole"), which shows up in drilling, completions, logging, cleanouts, and
fishing jobs alike — to show, with real numbers, how much the surrounding
context changes what the model predicts. No made-up rule maps "POOH" to
one category; you watch the real distribution shift as the sentence around
it changes.

**You'll be able to answer:** What is my sentence actually being turned
into before the model sees it? Why might the model's "obvious" next word
not be so obvious after all? Why does adding more context change the
answer?

### Notebook 2: Why can the same question get a different answer?

`notebooks/02_temperature_sampling_and_decoding_strategies.ipynb`

Takes the probabilities from notebook 1 and shows how one token actually
gets chosen and turned into generated text: **greedy decoding** (always
take the top answer — deterministic) versus **sampling** (draw randomly,
weighted by probability — so a lower-probability word can still come out).
Shows exactly what the "temperature" setting in any AI tool actually does
to those real numbers, and what "top-k" / "top-p" mean when you see them
in a model's settings.

**You'll be able to answer:** Why did I get a different answer when I
asked the same question twice? What does turning down an AI tool's
"temperature" actually do? Is the model "guessing," and if so, how?

## Installation

Works on **macOS, Windows, and Linux** — the steps are the same except for
one command when you activate the virtual environment.

1. Make sure you have Python 3.10+ installed.
2. From the repo root, create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

   (On Windows, if `python3` isn't recognized, use `python` instead.)

3. Activate it:

   **macOS / Linux:**
   ```bash
   source .venv/bin/activate
   ```

   **Windows (Command Prompt):**
   ```bat
   .venv\Scripts\activate
   ```

   **Windows (PowerShell):**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to run

```bash
jupyter notebook notebooks/01_how_a_real_llm_predicts_the_next_token.ipynb
jupyter notebook notebooks/02_temperature_sampling_and_decoding_strategies.ipynb
```

Run the cells in order, top to bottom, in either notebook — each stands on
its own. The first code cell that loads the model will download it from
Hugging Face the first time you run either one; after that it's cached and
reused.

## Model and hardware expectations

- **Model:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
  an open-source instruction-tuned model, run entirely locally via Hugging
  Face Transformers + PyTorch. No external API calls are made for
  inference — nothing you type leaves your machine.
- **Download size:** approximately 3 GB, downloaded once and cached in
  your user folder (`~/.cache/huggingface` on macOS/Linux;
  `C:\Users\<you>\.cache\huggingface` on Windows) — you won't need to
  manage this yourself, it's handled automatically.
- **Hardware:** works on Mac, Windows, or Linux. The notebook automatically
  detects and uses whichever is fastest on your machine — an Apple Silicon
  Mac's built-in GPU, an NVIDIA GPU (common on Windows and Linux
  desktops/laptops), or, if neither is present, your regular processor
  (CPU). No GPU is required — CPU-only execution works fine on any modern
  laptop, just somewhat slower.
- **Offline use:** after the first download, the notebook runs fully
  offline.
- If `Qwen2.5-1.5B-Instruct` fails to load in your environment for any
  reason, the notebook automatically falls back to a smaller sibling model
  and clearly reports which model actually ran — it will never silently
  substitute a model without telling you.

## What's in this repo

```
oilfield-llm-next-token-lab/
├── notebooks/                                    <- main path, start here
│   ├── 01_how_a_real_llm_predicts_the_next_token.ipynb
│   └── 02_temperature_sampling_and_decoding_strategies.ipynb
├── advanced/                                      <- optional, see advanced/README.md
│   ├── README.md
│   ├── 03_embeddings_and_attention.ipynb
│   ├── 04_gradient_attribution_and_occlusion.ipynb
│   ├── 05_activation_patching_and_causal_tracing.ipynb
│   ├── 06_probing_classifiers.ipynb
│   ├── 07_individual_head_circuit_analysis.ipynb
│   └── 08_individual_neuron_analysis.ipynb
├── requirements.txt
└── README.md
```

## What this project promises

Every probability, logit, and token shown in the main path is read
directly out of the loaded model — nothing is hard-coded, simulated, or
adjusted to produce a tidier-looking result. If the model's real answer
isn't the intuitive oilfield word, the notebooks show that honestly rather
than filtering it out. **No AI or programming background is required for
the main path** (notebooks 1–2). The advanced path assumes real ML/Python
fluency — see `advanced/README.md`.
