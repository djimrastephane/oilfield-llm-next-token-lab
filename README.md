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
| **Where** | `notebooks/` (5 notebooks) | `advanced/` (8 notebooks) |
| **Time** | ~70–90 minutes | Several hours |
| **Answers** | "What does the model actually do, and how do I use it well?" | "What's mechanistically happening inside, and how rigorously can we claim to know that?" |

**If you're an oil & gas professional, you only need the main path.** The
advanced path is real, technically rigorous interpretability research
(attention, gradient attribution, activation patching, probing
classifiers, individual attention heads and neurons) — genuinely
interesting, but not required to get practical value from this project.
See [`advanced/README.md`](advanced/README.md) for that path; the rest of
this README is about the main path.

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
4. Does a high probability mean the answer is correct?
        NO -- a fluent, confident, specific number can still be made up
   |
5. How do we make it safer for engineering use (grounding, RAG)?
        give the model the real document -> it reads instead of guesses
   |
6. What should an engineer remember, day to day?
        the four real findings above, turned into five practical rules
```

All six steps are built today, as `notebooks/01_...`, `notebooks/02_...ipynb`,
`notebooks/03_...ipynb`, `notebooks/04_...ipynb`, and `notebooks/05_...ipynb`.
The main path is complete — see below for what each notebook covers.

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

### Notebook 3: Does a high probability mean the answer is correct?

`notebooks/03_does_high_probability_mean_correct.ipynb`

The most consequential lesson in the series for actual work use. Watches
the real model write a confident-sounding field note with a specific
pressure test value for a completely fictional well, then runs the real
test that matters: does that number change when nothing relevant about
the question does? (It does — real result, shown honestly.) Then checks
whether this only happens with made-up scenarios, using a genuinely
well-established industry fact (one barrel of oil = 42 US gallons) —
and finds that even that can come out wrong depending only on how the
question is phrased.

**You'll be able to answer:** Can an AI tool sound completely confident
and still be wrong? How would I even test that, myself, on my own
questions? What should I actually verify before trusting a specific
number an AI tool gives me?

### Notebook 4: Grounding answers in real documents (RAG)

`notebooks/04_grounding_answers_in_real_documents.ipynb`

The fix for what notebook 3 found. Instead of asking the real model to
recall a fact, this notebook shows it a real reference document and lets
it read the answer off the page instead — and proves, live, that this
turns a guess into a correct, checkable answer. Then it builds a small,
fully-visible retrieval step (no hidden "embeddings" — just counting
shared words) that automatically finds the right document out of several,
wires it together into a real, working retrieve-then-answer pipeline
(RAG), and honestly tests two ways it can still go wrong: being handed
the *wrong* document (the model repeats the wrong number just as
confidently), and having *no* matching document at all (compares a plain
prompt against one that explicitly tells the model to admit when
information is missing, and reports what actually happened rather than
assuming).

**You'll be able to answer:** If the model can't be trusted to recall a
fact, can it still get it right when the fact is put in front of it? Where
does the right document actually come from? Does grounding guarantee a
correct answer, or can it still fail — and how?

This notebook's retrieval step deliberately uses simple word-counting,
not "embeddings," so every step stays visible. If you're curious how a
real meaning-based (embedding) search actually performs on oilfield
language instead, see the advanced path's
[`03_embeddings_and_attention.ipynb`](advanced/03_embeddings_and_attention.ipynb)
— it tests that directly on this same model, with real results, including
where it falls short.

### Notebook 5: Day-to-day rules for using an AI tool at work

`notebooks/05_day_to_day_rules_for_using_ai_tools.ipynb`

The capstone. No new theory — it pulls the four real findings from
notebooks 1-4 together into five practical rules, then applies the last
three of them together, live, to a fresh question this series has never
used before (a torque specification): checking whether an ungrounded
answer is even consistent across reworded questions, then grounding it
with the same real retrieve-then-answer pipeline from notebook 4. Ends
with a one-page, five-rule summary you can actually keep next to your
desk, and closes out the main path.

**You'll be able to answer:** Given everything the first four notebooks
showed, what should I actually *do* differently the next time an AI tool
gives me an answer at work?

## How to run: three ways, pick what fits you

### Easiest: open in Google Colab — no installation at all

Click a badge, and the notebook opens and runs in your browser on a free
Google server. Nothing to install on your own computer.

[![Open Notebook 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimrastephane/oilfield-llm-next-token-lab/blob/main/notebooks/01_how_a_real_llm_predicts_the_next_token.ipynb)
**Notebook 1** — What does an LLM actually do?

[![Open Notebook 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimrastephane/oilfield-llm-next-token-lab/blob/main/notebooks/02_temperature_sampling_and_decoding_strategies.ipynb)
**Notebook 2** — Why can the same question get a different answer?

[![Open Notebook 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimrastephane/oilfield-llm-next-token-lab/blob/main/notebooks/03_does_high_probability_mean_correct.ipynb)
**Notebook 3** — Does a high probability mean the answer is correct?

[![Open Notebook 4 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimrastephane/oilfield-llm-next-token-lab/blob/main/notebooks/04_grounding_answers_in_real_documents.ipynb)
**Notebook 4** — Grounding answers in real documents (RAG)

[![Open Notebook 5 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimrastephane/oilfield-llm-next-token-lab/blob/main/notebooks/05_day_to_day_rules_for_using_ai_tools.ipynb)
**Notebook 5** — Day-to-day rules for using an AI tool at work

Every notebook above was tested end to end on Colab's free tier before
being added here: on a T4 GPU runtime, downloading the ~3 GB model took
under a minute (Colab's connection is much faster than a typical home
connection), and each notebook ran without any errors or code changes —
notebooks 4 and 5's real, live outputs on Colab's GPU matched the ones
from a local run exactly. A few things to expect, honestly:

- The first time you open the link, Colab shows a one-time warning that
  the notebook wasn't authored by Google, since it's loading from GitHub.
  Click **"Run anyway"** — you're looking at the real, public source on
  GitHub, and can review it there first if you'd like.
- For the best speed, use the menu **Runtime → Change runtime type →
  T4 GPU** (free) before running — the notebook works on the default
  CPU setting too, just slower.
- Colab's copy of the model isn't saved between sessions, so it
  re-downloads (quickly) each time you open a fresh Colab session.
- Nothing you type is private here the way it is with local execution —
  your inputs run on a Google-hosted server, not your own machine. For
  anything sensitive, use the local option below.

### Local: run everything on your own computer

Nothing you type leaves your machine — the most private option, and the
one used for every real result in this README. The setup below has the
same steps on **macOS, Windows, and Linux**, except for one command when
you activate the virtual environment — but honestly, every real result in
this README was actually generated on macOS specifically. Windows and
Linux run the identical code path (standard Python, PyTorch, and Jupyter,
with no OS-specific logic beyond that one activation command), but
haven't been independently re-verified end to end on this project's side.
If you hit a platform-specific snag, please open a GitHub issue.

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

5. Run a notebook:

   ```bash
   jupyter notebook notebooks/01_how_a_real_llm_predicts_the_next_token.ipynb
   jupyter notebook notebooks/02_temperature_sampling_and_decoding_strategies.ipynb
   jupyter notebook notebooks/03_does_high_probability_mean_correct.ipynb
   jupyter notebook notebooks/04_grounding_answers_in_real_documents.ipynb
   jupyter notebook notebooks/05_day_to_day_rules_for_using_ai_tools.ipynb
   ```

Run the cells in order, top to bottom, in any notebook — each stands on
its own. The first code cell that loads the model will download it from
Hugging Face the first time you run either one; after that it's cached and
reused.

### Advanced: clone and configure manually

For the `advanced/` interpretability notebooks, or if you want full
control over your environment:

```bash
git clone https://github.com/djimrastephane/oilfield-llm-next-token-lab.git
cd oilfield-llm-next-token-lab
```

Then follow the **Local** steps above. See
[`advanced/README.md`](advanced/README.md) for what each of those eight
notebooks needs.

## Model and hardware expectations

- **Model:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
  an open-source instruction-tuned model, run entirely locally via Hugging
  Face Transformers + PyTorch. No external API calls are made for
  inference — nothing you type leaves your machine.
- **This specific model isn't a requirement.** It's simply what this
  tutorial was built and verified against, chosen because it's small
  enough to run on an ordinary laptop. Nothing in the mechanics taught
  here (tokenization, softmax, decoding, grounding, and so on) is
  specific to this one model — you're free to point `PRIMARY_MODEL_NAME`
  at a different Hugging Face model your hardware can handle. Just don't
  expect identical numbers: next-token probabilities depend on what a
  model was actually trained on, so a different model can reasonably
  produce a different top candidate, a different generated sentence, or a
  different guess for notebook 3's fictional well. That's not a bug —
  it's the same lesson these notebooks teach, playing out on a new model.
- **Download size:** approximately 3 GB, downloaded once and cached in
  your user folder (`~/.cache/huggingface` on macOS/Linux;
  `C:\Users\<you>\.cache\huggingface` on Windows) — you won't need to
  manage this yourself, it's handled automatically.
- **Hardware:** the notebook automatically detects and uses whichever is
  fastest on your machine — an Apple Silicon Mac's built-in GPU, an
  NVIDIA GPU (common on Windows and Linux desktops/laptops), or, if
  neither is present, your regular processor (CPU) — on any of macOS,
  Windows, or Linux (see the note above on what's actually been tested
  on each). No GPU is required — CPU-only execution works fine on any
  modern laptop, just somewhat slower.
- **Offline use:** after the first download, the notebook runs fully
  offline.
- If `Qwen2.5-1.5B-Instruct` fails to load in your environment for any
  reason, the notebook automatically falls back to a smaller sibling model
  (`Qwen2.5-0.5B-Instruct`, about 950 MB) and clearly reports which model
  actually ran — it will never silently substitute a model without
  telling you. This isn't just a description of the code: the fallback
  was deliberately triggered and confirmed working end to end before this
  claim was written.

## What's in this repo

```
oilfield-llm-next-token-lab/
├── notebooks/                                    <- main path, start here
│   ├── 01_how_a_real_llm_predicts_the_next_token.ipynb
│   ├── 02_temperature_sampling_and_decoding_strategies.ipynb
│   ├── 03_does_high_probability_mean_correct.ipynb
│   ├── 04_grounding_answers_in_real_documents.ipynb
│   └── 05_day_to_day_rules_for_using_ai_tools.ipynb
├── advanced/                                      <- optional, see advanced/README.md
│   ├── README.md
│   ├── 03_embeddings_and_attention.ipynb
│   ├── 04_gradient_attribution_and_occlusion.ipynb
│   ├── 05_activation_patching_and_causal_tracing.ipynb
│   ├── 06_probing_classifiers.ipynb
│   ├── 07_individual_head_circuit_analysis.ipynb
│   ├── 08_individual_neuron_analysis.ipynb
│   ├── 09_grounding_answers_in_real_documents.ipynb
│   └── 10_day_to_day_rules_for_using_ai_tools.ipynb
├── requirements.txt
└── README.md
```

## What this project promises

Every probability, logit, and token shown in the main path is read
directly out of the loaded model — nothing is hard-coded, simulated, or
adjusted to produce a tidier-looking result. If the model's real answer
isn't the intuitive oilfield word, the notebooks show that honestly rather
than filtering it out. **No AI or programming background is required for
the main path** (notebooks 1–5) — the Colab option above means you don't
even need to install anything to see that for yourself. The advanced path
assumes real ML/Python fluency — see [`advanced/README.md`](advanced/README.md).
