# mlfeel

A sentiment-analysis engine built from scratch on top of NumPy. No `scikit-learn`, no
`torch`: the text pipeline, the autodiff-style computation graph, the optimizer, and the
loss functions are all implemented by hand. The goal is to reduce sentiment classification
to its atomic components and keep every layer auditable.

## Status

The training core is complete and validated by Pyright (strict typing) and the test suite.
On a small, linearly separable toy corpus the model converges to 0 loss and 100% accuracy.

## Architecture

The codebase is split into independent, single-responsibility modules under `src/mlfeel/`.

```
text -> BaseTokenizer -> VocabIndexer -> TF_IDF_Vectorizer -> Sequential(MLP) -> logits
```

| Module | Responsibility |
| --- | --- |
| `tokenizer.py` | `BaseTokenizer`: deterministic RegEx normalization (lowercase, HTML strip, non-alphanumeric removal, whitespace collapse) and tokenization. |
| `vocabulary.py` | `VocabIndexer`: builds the bijective `word_to_index` / `index_to_word` tables. Vocabulary is sorted before indexing to guarantee reproducible column ordering. |
| `vectorizer.py` | `TF_IDF_Vectorizer`: `fit` computes the IDF vector, `transform` produces the TF-IDF matrix. Term frequency is normalized per document; IDF uses smoothed `log((1+N)/(1+df)) + 1`. |
| `model.py` | `Module` base class, `Linear`, `Sequential`, and the `train_model` loop (mini-batch SGD with shuffling). |
| `activations/` | `Activation` base class and `LeakyReLU`. |
| `losses/` | `Loss` base class, `CrossEntropy` (stable softmax + analytical gradient), `MSE`. |
| `optimizers/` | `Optimizer` base class and `Adam` (bias-corrected moments). |

### Design notes

- **Module contract.** Every layer (`Linear`, `Activation`, `Sequential`) implements the
  same `forward` / `backward` / `parameters` interface. This lets `Sequential` chain
  arbitrary modules and run backprop generically by reversing the layer list. Activations
  share the same `Module` shape, so the graph stays uniform.

- **Parameters as references.** `parameters()` returns `[{"param", "grad"}]` dictionaries
  holding references to the actual arrays. `Adam.step` mutates parameters in place
  (`parameter -= ...`) and keys its optimizer state on `id(parameter)`. This keeps the
  optimizer decoupled from the network topology: it never needs to know the model layout,
  only the flat list of parameter/gradient pairs.

- **Fused softmax + cross-entropy.** `CrossEntropy._deriv` returns `(P - Y) / N` directly
  with respect to the raw logits, avoiding a separate softmax layer in the backward pass
  and the numerical instability that comes with it. The forward pass subtracts the per-row
  max for a stable softmax and clips probabilities before the log.

- **Vectorizer is part of the model.** The `word_to_index` table and the `idf` vector are
  fitted state, not hyperparameters. At inference time the same vectorizer instance must be
  reused; re-fitting on new data would shift column indices and break the first `Linear`
  layer's input mapping.

- **Determinism.** Sorted vocabulary, seeded RNG in tests. Same input, same weights.

## Project layout

```
src/mlfeel/        engine modules
tests/             pytest suite (one file per module, plus an end-to-end sentiment test)
main.py            placeholder entry point
.github/workflows  CI: ruff + pytest on push / PR to main
```

## Usage

```bash
uv sync                 # install dependencies
uv run pytest           # run the test suite
uv run ruff check .     # lint
```

End-to-end training (see `tests/test_sentiment.py`):

```python
vectorizer = TF_IDF_Vectorizer()
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus).astype(np.float32)

model = Sequential([
    Linear(vectorizer.dim, 16),
    LeakyReLU(alpha=0.01),
    Linear(16, 2),
])
loss_fn = CrossEntropy()
optimizer = Adam(learning_rate=0.05, beta_momentum=0.9, beta_rms=0.999)

train_model(model, loss_fn, optimizer, X_train, y_train, epochs=200, batch_size=4)
```

## Roadmap

### Serialization
- Export a single artifact bundling `word_to_index`, the `idf` vector, and all
  `Sequential` parameter matrices. Loading network weights without the matching vectorizer
  state is the main production trap.
- Add `save` / `load` methods (pickle or a structured binary format) with a versioned
  format header.

### Inference serving (FastAPI)
- `POST /predict`: accept a raw string, run `vectorizer.transform([text])` (list to keep
  the `N=1` dimension), then `model.forward`, then `argmax` on the logits.
- Load the artifact once at startup; the route never calls `train_model` or
  `optimizer.step`.
- Health/readiness endpoints and input validation.

### DevOps
- Dockerfile and container build in CI.
- Extend the CI matrix beyond Python 3.11.
- Add `pyright` as a CI gate alongside ruff.

### Model polymorphism
- Pluggable architectures behind the existing `Module` interface (extra layers, dropout,
  alternative activations).
- Train on a real dataset (e.g. IMDB) rather than the toy corpus, with a held-out
  validation split.

### Energy / Green AI
- Measure joules per inference. On the current macOS (M4 Pro) CPU target, estimate
  theoretical cost from matrix-multiply FLOPs (2x FLOPs) and compare against measured
  consumption.
- Reference methodologies: CodeCarbon, RAPL counters (Linux), Apple performance counters.
