# Guide: Multimodal — Vision, Audio, Text, Graphs, Time Series

*Beyond tabular data: 100+ functions across five feature suites.*

> Back to [Guides index](README.md) · Model choices: [MODEL_CATALOG.md](MODEL_CATALOG.md) (vision, speech sections)

---

## What you'll learn

The five multimodal sub-packages and their most-used functions:

| Package | Domain | Highlights |
|---|---|---|
| `qai.vision` | images | filters, color ops, feature extraction |
| `qai.audio` | sound | spectrograms, MFCCs, zero-crossing |
| `qai.nlp` | text | tokenization, n-grams, stemming, distance |
| `qai.graph` | networks | centralities, traversal |
| `qai.time_series` | sequences | decomposition, statistics |

---

## 1. Vision (`qai.vision`)

```python
from qai import vision

# Transforms
img = vision.resize(img, (224, 224))
img = vision.crop(img, box=(0, 0, 100, 100))
img = vision.rotate(img, degrees=15)
img = vision.flip(img, horizontal=True)
img = vision.grayscale(img)

# Enhancement
img = vision.blur(img, radius=2)
img = vision.sharpen(img)
img = vision.adjust_brightness(img, 1.2)
img = vision.adjust_contrast(img, 1.1)

# Feature extraction
edges = vision.edge_detect(img)          # spatial filters
feat = vision.extract_features(img)      # downstream model input
```

### Full media pipeline (`qai.MediaManipulator`)

Declarative, batchable, with an audit trail — 15 tests:

```python
from qai import MediaManipulator

mm = MediaManipulator()
mm.resize(224, 224).grayscale().sharpen()
out = mm.apply("photo.jpg")             # single
outs = mm.apply_batch(image_paths)      # batch
print(mm.audit_trail())                 # every op recorded
```

Model choices for vision: `vit`, `deit`, `resnet`, `densenet`, `efficientnet`,
`mobilenet`, `unet`, `clip`, `nerf`, `inr` — see
[MODEL_CATALOG.md §7](MODEL_CATALOG.md).

CNN input handling is forgiving: `(H,W)`, `(H,W,C)`, `(N,H,W)`, `(N,H,W,C)`,
`(N,C,H,W)` and lists all work; uint8 inputs are auto-normalized.

---

## 2. Audio (`qai.audio`)

```python
from qai import audio

spec = audio.spectrogram(samples)           # power spectrogram
mfcc = audio.mfcc(samples, n_mfcc=13)       # classic speech features
zcr  = audio.zero_crossing_rate(samples)    # noisiness indicator
centroid = audio.spectral_centroid(samples) # brightness indicator

feat = audio.extract_features(samples)      # combined feature vector
```

Model choices for speech: `whisper` (speech → text), `wav2vec2`, `hubert` —
[MODEL_CATALOG.md §7](MODEL_CATALOG.md).

---

## 3. Text (`qai.nlp`)

```python
from qai import nlp

tokens = nlp.word_tokenize(text)
ngrams = nlp.ngrams(text, n=2)
stems  = nlp.stem(words)                    # lightweight stemmer

sim  = nlp.cosine_similarity(vec_a, vec_b)
dist = nlp.levenshtein_distance("kitten", "sitting")   # edit distance
```

### Real tokenizers (`qai.tokenizers`)

```python
from qai.tokenizers import CharTokenizer, WordTokenizer, BPETokenizer

tok = BPETokenizer(vocab_size=1000)
tok.fit(corpus)

ids  = tok.encode("hello world")            # → token ids
txt  = tok.decode(ids)                       # round-trip
batch = tok.encode_batch(list_of_texts)      # padded batch
tok.save("tok.json"); tok2 = BPETokenizer.load("tok.json")
```

All three tokenizers share the API, include the four special tokens
(`<pad>`, `<unk>`, `<bos>`, `<eos>`), and support padding, truncation, and
BOS/EOS wrapping. Vectorizer alternatives: `count_vectorizer`,
`tfidf_vectorizer`, `hashing_vectorizer` (all in the technique registry).

---

## 4. Graphs (`qai.graph`)

```python
from qai import graph

deg  = graph.degree_centrality(G)
btw  = graph.betweenness_centrality(G)
close = graph.closeness_centrality(G)
pg   = graph.pagerank(G)

path = graph.shortest_path(G, source, target)
comps = graph.connected_components(G)
```

Model choices: `gnn`, `gcn`, `graphsage`, `gat`, `gin`, `mpnn` —
[MODEL_CATALOG.md §10](MODEL_CATALOG.md).

---

## 5. Time series (`qai.time_series`)

```python
from qai import time_series as ts

trend, seasonal, resid = ts.decompose(series, period=7)
stats = ts.statistics(series)               # mean, variance, autocorrelation...
feats = ts.extract_features(series)         # model-ready feature vector

# Forecasting models:
#   qai.build(technique="time_series").forecast(steps=30)
#   qai.build(technique="time_series_arima")
#   qai.build(technique="kalman_filter")
```

Feature engineering in `qai.data`: `rolling_window(series, window=7)`,
`lag_feature(series, lag=1)` — see [DATA.md §5](DATA.md).

---

## Cheat sheet

| Task | Call |
|---|---|
| Image pipeline | `MediaManipulator().resize().grayscale().apply(path)` |
| Batch images | `mm.apply_batch(paths)` |
| Speech features | `qai.audio.mfcc(samples)` |
| Text distance | `qai.nlp.levenshtein_distance(a, b)` |
| Trainable tokenizer | `qai.tokenizers.BPETokenizer(vocab_size=1000)` |
| Graph centrality | `qai.graph.pagerank(G)` |
| Decompose series | `qai.time_series.decompose(series, period)` |
| Forecast | `qai.build(technique="time_series").forecast(steps)` |
| Vision models | `vit / resnet / efficientnet / clip / sam2` |
| Speech models | `whisper / wav2vec2 / hubert` |

---

## Common gotchas

- **Image channel order:** CNN auto-detects, but consistent `(N,C,H,W)` is fastest.
- **MFCC assumes 16kHz-ish sample rates** — resample first if yours differ wildly.
- **BPE needs fitting.** Unfit tokenizers have no merges — call `fit(corpus)` before `encode`.
- **Graph functions expect adjacency dicts/matrices** — check each function's accepted format in the [User Manual](../USER_MANUAL.md).

---

*Next: [MLOPS.md](MLOPS.md) or back to [Guides index](README.md).*
