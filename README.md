# Music Recommender with Semantic Scoring

The original project was a music recommender simulation from Module 4 which acted like a small music recommender system similar to applications like Spotify. It mirrored real world AI recommenders and represented songs and a user "taste profile" as data, along with other features.

[Demo on Hugging Face](https://huggingface.co/spaces/linda14/Music-Recommender )
[Video Walkthrough](https://youtu.be/KY2S7lffhN4)
[Presentation](https://canva.link/ifhjo9unf6e81yy)
---
title: Music Recommender Simulation
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.41.0"
app_file: app.py
pinned: false
---


### Terminal Image Output
<a><img src="/assets/Output.png" alt="Terminal Output" width="600"/></a>

## Title and Summary

This project is a content-based music recommender that uses semantic similarity to match songs to a user's taste profile. Instead of asking whether a user's genre preference is an exact string match to a catalog entry, it embeds both the user's preference and every song's genre and mood into a shared vector space using the `all-MiniLM-L6-v2` sentence-transformers model, then scores by cosine similarity.

**Why it matters:** The original system silently failed anyone whose preferred genre didn't appear word-for-word in the catalog, a k-pop fan reached a maximum score of 2.0/4.0 while a pop fan could reach 4.0, purely because of string inequality. Semantic scoring closes that gap by giving partial credit when genres are meaningfully related, and it makes the scoring logic honest about what it can and cannot do.


---

## Architecture Overview

The full system is documented in [docs/system_diagram.md](docs/system_diagram.md). The short version:

```
User Profile ──┐
               ▼
data/songs.csv → load_songs() → embed_catalog() → score_song() × 20 → recommend_songs() → Ranked Top-5
                                    ▲
                          all-MiniLM-L6-v2
                         (loads once at startup)
```

There are six distinct components:

| Component | Role |
|---|---|
| **Catalog Loader** (`load_songs`) | Parses `data/songs.csv` into typed dicts; validates and logs errors |
| **Semantic Embedder** (`build_embedder`, `embed_catalog`) | Loads the sentence-transformers model once and attaches pre-computed numpy vectors to each song |
| **Semantic Scorer** (`score_song`) | Replaces `==` for genre and mood with cosine similarity; keeps energy as a continuous distance; falls back to exact-match if the model is unavailable |
| **Ranker** (`recommend_songs`) | Embeds the user profile once per query, scores all 20 songs, sorts descending, returns top-k with explanation strings |
| **Human-in-the-loop** | `src/main.py` runs 6 profiles (3 normal, 3 adversarial) and prints score breakdowns for human inspection |
| **Test Suite** (`pytest`) | Two tests validate sort order and explanation format via the exact-match fallback path |

---

### Mermaid Diagram of System

<a><img src="assets/Song Recommendation Scoring.png" alt="System Diagram" width="800"/></a>

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Internet connection on first run (model download, ~80 MB, cached after that)

### Steps

1. **Clone the repository and navigate to the project folder.**

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   On first run, `sentence-transformers` will automatically download the
   `all-MiniLM-L6-v2` model from HuggingFace (~80 MB). Subsequent runs use
   the local cache and require no internet connection. If the download fails,
   the app logs a warning and continues with exact-match scoring.

4. **Run the recommender:**

   ```bash
   python -m src.main
   ```

5. **Run the tests:**

   ```bash
   pytest
   ```

   Expected output: `2 passed`.

---

## Sample Interactions

All examples below are taken from live output of `python -m src.main`.

### Example 1: Normal profile: High-Energy Pop

**Input**
```
favorite_genre: pop   preferred_mood: happy   target_energy: 0.80
```

**Output (top 3 of 5)**
```
#1  Sunrise City  —  Neon Echo
    Genre: pop  |  Mood: happy  |  Energy: 0.82
    Score: 3.98 / 4.00
    + genre similarity 1.00 (pop → pop) +2.00
    + mood similarity 1.00 (happy → happy) +1.00
    + energy similarity 0.98 (target 0.8, song 0.82)

#2  Rooftop Lights  —  Indigo Parade
    Genre: indie pop  |  Mood: happy  |  Energy: 0.76
    Score: 3.23 / 4.00
    + genre similarity 0.64 (indie pop → pop) +1.27
    + mood similarity 1.00 (happy → happy) +1.00
    + energy similarity 0.96 (target 0.8, song 0.76)

#3  Gym Hero  —  Max Pulse
    Genre: pop  |  Mood: intense  |  Energy: 0.93
    Score: 3.20 / 4.00
    + genre similarity 1.00 (pop → pop) +2.00
    + mood similarity 0.33 (intense → happy) +0.33
    + energy similarity 0.87 (target 0.8, song 0.93)
```

**What this shows:** Rooftop Lights (indie pop) reaches #2 with a partial genre credit of +1.27. Under the old exact-match system it earned zero genre points and could never rank this high. The explanation line `(indie pop → pop) +1.27` makes the partial credit visible rather than hiding it.

---

### Example 2 : Adversarial profile: Ghost Genre (k-pop)

**Input**
```
favorite_genre: k-pop   preferred_mood: happy   target_energy: 0.75
```
*(k-pop does not appear anywhere in the catalog)*

**Output (top 2 of 5)**
```
#1  Rooftop Lights  —  Indigo Parade
    Genre: indie pop  |  Mood: happy  |  Energy: 0.76
    Score: 3.24 / 4.00
    + genre similarity 0.62 (indie pop → k-pop) +1.25
    + mood similarity 1.00 (happy → happy) +1.00
    + energy similarity 0.99 (target 0.75, song 0.76)

#2  Sunrise City  —  Neon Echo
    Genre: pop  |  Mood: happy  |  Energy: 0.82
    Score: 3.04 / 4.00
    + genre similarity 0.56 (pop → k-pop) +1.11
    + mood similarity 1.00 (happy → happy) +1.00
    + energy similarity 0.93 (target 0.75, song 0.82)
```

**What this shows:** Under exact-match scoring this user was capped at 2.0/4.0 (zero genre bonus forever). With semantic scoring they reach 3.24/4.00, and the #1 result is a genuinely appropriate recommendation, an upbeat, high-energy pop-adjacent track with a matching mood.

---

### Example 3: Adversarial profile: Energy-Mood Conflict

**Input**
```
favorite_genre: blues   preferred_mood: sad   target_energy: 0.90
```
*(requesting high energy, but blues/sad only exists as a slow, quiet song)*

**Output (top 2 of 5)**
```
#1  Last Train South  —  Earl Dusty
    Genre: blues  |  Mood: sad  |  Energy: 0.38
    Score: 3.48 / 4.00
    + genre similarity 1.00 (blues → blues) +2.00
    + mood similarity 1.00 (sad → sad) +1.00
    + energy similarity 0.48 (target 0.9, song 0.38)

#2  Storm Runner  —  Voltline
    Genre: rock  |  Mood: intense  |  Energy: 0.91
    Score: 2.11 / 4.00
    + genre similarity 0.43 (rock → blues) +0.86
    + mood similarity 0.27 (intense → sad) +0.27
    + energy similarity 0.99 (target 0.9, song 0.91)
```

**What this shows:** The genre+mood bonuses (3.0 combined) still outweigh the severe energy mismatch (0.48 vs a possible 1.0), so Last Train South ranks first even though its energy is wrong by 0.52. This is an honest, documented limitation, the system works correctly by its own rules but those rules don't fully capture what this user actually wants.

---

## Design Decisions

**Sentence-transformers over an API call.**
The embedder runs locally with no API key and no per-call cost. `all-MiniLM-L6-v2` is 80 MB, downloads once, and handles genre vocabulary (indie pop, k-pop, lofi, synthwave) with good proximity. Using an API-based embedder would add latency, cost, and a network dependency to every run.

**Embed genre and mood separately, not as one string.**
The original scoring recipe weights genre at 2× and mood at 1×. Concatenating them into one embedding would collapse that distinction. Keeping them separate lets the weight ratio survive the move to semantic scoring.

**Pre-compute catalog embeddings at startup, embed user profile once per query.**
With 20 songs and 6 profiles, this means 40 catalog vectors computed once, plus 12 user-profile vectors computed at query time, rather than 20 × 6 = 120 embedding calls in the naive approach. At this scale the difference is small, but the pattern is correct for larger catalogs.

**Graceful fallback to exact-match.**
If `sentence-transformers` fails to import or the model download fails, the scorer falls back to `==` comparison and logs a warning. The app never crashes on a missing dependency, and the test suite exercises this path by passing plain `Song` objects with no embeddings attached.

**Keep energy as a continuous distance, not an embedding.**
Energy is a float in [0, 1]. It doesn't need vector semantics, the formula `1 − |song.energy − target_energy|` is already a meaningful similarity measure with a clear interpretation. Replacing it with an embedding would add complexity without improving accuracy.

**Trade-off acknowledged: genre+mood can still override energy.**
The 2:1:1 weight ratio means a perfect genre+mood match scores 3.0 before energy is even counted. A song with completely wrong energy (0.48 when target is 0.90) can still rank first if genre and mood match perfectly. This is a known limitation, documented in the adversarial profiles. Fixing it would require rethinking the weight structure, not just the similarity function.

---

## Reliability Testing

Two methods were used to verify the system works as claimed, not just appears to work.

### Method 1 — Automated Unit Tests

Seven tests in [tests/test_recommender.py](tests/test_recommender.py) cover both the fallback path (exact-match) and the semantic scoring path. The five new reliability tests use mock numpy vectors — no model download needed — so they run in under a second and prove the math is correct independently of the embedding model.

| Test | What it proves |
|---|---|
| `test_recommend_returns_songs_sorted_by_score` | Sort order is correct for a pop/happy user |
| `test_explain_recommendation_returns_non_empty_string` | Explanation strings are always produced |
| `test_cosine_sim_identical_vectors_returns_one` | `_cosine_sim` returns 1.0 for identical vectors |
| `test_cosine_sim_orthogonal_vectors_returns_zero` | `_cosine_sim` returns 0.0 for orthogonal vectors |
| `test_score_song_perfect_embedding_match_scores_four` | A perfect genre + mood + energy alignment scores exactly 4.0 |
| `test_ghost_genre_gets_partial_credit_not_zero_not_full` | A song at a 45° embedding angle earns > 0 and < 2.0 genre points |
| `test_recommend_songs_returns_k_results_in_descending_order` | Top-k slice is always sorted highest-first |

**Result: 7 out of 7 tests passed.** The AI scored correctly across all boundary conditions tested. The one known failure mode — genre+mood bonuses overriding a severe energy mismatch (Profile 4) — is not a code bug; it is a documented weight-design choice that the tests intentionally do not paper over.

Run with:
```bash
pytest -v
```

### Method 2 — Logging and Error Handling

Every key decision the system makes is written to stdout via Python's `logging` module at `INFO` level. This makes failures and fallbacks observable without reading the source code.

Examples of what gets logged:

```
INFO  Loaded 20 songs from 'data/songs.csv'.
INFO  Loading sentence-transformers model 'all-MiniLM-L6-v2'...
INFO  Model loaded successfully.
INFO  Computing embeddings for 20 songs...
INFO  Catalog embeddings ready. Semantic scoring is active.
WARNING  Could not load sentence-transformers (...). Falling back to exact-match scoring.
WARNING  Energy value 1.30 for 'Bad Song' is out of [0, 1]; clamping.
ERROR  Catalog file not found: data/songs.csv
```

The fallback warning in particular proves that the app never silently degrades — if semantic scoring is unavailable, a human reading the log knows immediately and can judge whether the output is still trustworthy.

---

## Testing Summary

**7 / 7 automated tests passed.** The semantic scorer correctly handles perfect matches (4.0), partial matches (0 < score < 2.0), and orthogonal mismatches (0.0). The ghost-genre ceiling problem (k-pop user capped at 2.0/4.0) is resolved — semantic scoring raised the reachable score to 3.24/4.00 and surfaced genuinely relevant songs. The one remaining documented failure is Profile 4 (energy-mood conflict): genre+mood similarity still totals 3.0, enough to rank a slow blues song above every high-energy song for a user who explicitly asked for high energy. That is a weight-structure issue, not a similarity-function issue, and it is visible in both the score breakdown and the logs.

### What the adversarial profiles revealed

| Profile | Finding |
|---|---|
| k-pop (ghost genre) | Semantic scoring raised the ceiling from 2.0 → 3.24/4.00 and surfaced genuinely relevant songs |
| Energy-Mood Conflict | Genre+mood bonuses still dominate; a slow blues song ranks #1 for a high-energy user — the 2:1:1 weight structure is the root cause, not the similarity function |
| Ignored Dimensions | Semantics improved genre/mood resolution but acousticness, valence, and tempo remain unscored — a perfect 4.0 is still achievable without the system ever reading those fields |

### What would come next

Adding semantically-scored acousticness and valence, expanding the catalog to 5–10 songs per genre, and a diversity rule preventing the same genre from dominating a single recommendation list would address the three remaining weaknesses without changing the core architecture.

---

## Critical Reflection

### What are the limitations or biases in your system?

**Weight imbalance.** The 2:1:1 ratio (genre : mood : energy) means that a perfect genre and mood alignment contributes 3.0 points before energy is considered at all. A song that is wrong on energy by 0.52 units can still rank #1 if it matches genre and mood exactly — Profile 4 (energy-mood conflict) demonstrates this precisely. Semantic scoring made genre and mood signals more accurate, but did not change their relative dominance.

**Catalog representation bias.** The 20-song catalog covers 17 genres, but most genres have exactly one song. A lofi fan has three songs competing for the top spots; a blues fan has one. This means the genre bonus behaves like a hard filter for most users — whoever wins the genre match wins the recommendation, with no intra-genre competition to rank by.

**Three fields silently ignored.** Acousticness, valence, and tempo are collected in the user profile and stored on every song, but none of them affect the score. A user who carefully specifies they want highly acoustic music gets the same result as one who did not — and the system gives no indication that their input was unused.

**Embedding model trained on general text.** `all-MiniLM-L6-v2` was not trained on music metadata. Its genre proximity scores reflect patterns in general English writing about music, not how listeners actually experience genre similarity. The model rates "hip-hop → k-pop" similarity at 0.58 and "pop → k-pop" at 0.56 — a defensible result, but one that should be treated as an approximation, not ground truth.

---

### Could your AI be misused, and how would you prevent that?

At the scale of this classroom project the risks are low, but the patterns are worth naming.

**Structural underscoring of non-Western music.** A recommender that uses genre labels from a Western-centric catalog will consistently surface lower scores for users whose preferred genres (K-pop, Afrobeats, Amapiano, Bollywood) are underrepresented or absent. Even with semantic scoring, a user whose genre sits far from the catalog's center of mass is penalized relative to one whose genre appears three times. At production scale this compounds — users who receive weaker recommendations engage less, which generates fewer training signals, which makes future models even less accurate for those users. Prevention requires catalog diversity by design, not as an afterthought.

**False confidence from high scores.** A score of 3.9/4.0 looks authoritative. A user who doesn't read the explanation strings might trust that score even though it was built on only 3 of the 6 profile fields they provided. Prevention: the explanation strings already make this visible, but a more explicit warning — "Note: acousticness, valence, and tempo were not used in this score" — would be more honest.

---

### What surprised you while testing your AI's reliability?

Two things were unexpected.

First, **"hip-hop → k-pop" scored higher than "pop → k-pop"** in the sentence-transformers embedding space (0.58 vs 0.56). The intuition would be that pop is closer to k-pop than hip-hop is. But the model has apparently learned that hip-hop and k-pop share production vocabulary (trap beats, idol group framing) in a way that surfaces in English text. This is not wrong, but it was not anticipated, and it is a reminder that embedding similarity reflects corpus statistics rather than a listener's lived experience of genre.

Second, **Profile 4 failed in exactly the same way after semantic scoring as it did before.** The expectation going in was that improving genre/mood similarity might soften the energy-mood conflict problem. It did not. The slow blues song still ranked #1 with a 3.48 score. Semantic scoring is not a general fix for weight imbalance — it only fixes the cases where string inequality was the barrier. Where the weights themselves are miscalibrated, a better similarity function makes no difference.

---

### Collaboration with Claude Code

This project was built in direct conversation with Claude Code throughout. The planning phase, implementation, testing, and documentation were all developed iteratively — the human directed the goals and made the architectural decisions; Claude Code generated code, caught edge cases, and pushed back when a framing was imprecise.

**One instance where the human gave genuinely useful direction:** Early in planning, the enhancement was described as "adding RAG." Claude Code flagged that what was actually being described — embed text fields, replace `==` with cosine similarity, use the user profile as the query — is not RAG in the technical sense. RAG means: embed documents, retrieve top-k, feed to an LLM for generation. None of that generation step was needed here. The human accepted that correction immediately. That single clarification prevented a significant scope expansion: no vector database, no LLM generation step, no prompt engineering — just 30 lines of numpy and a sentence-transformers model. The right label led directly to the right solution size.

---

## Project Structure

```
.
├── data/
│   └── songs.csv               # 20-song catalog
├── docs/
│   ├── system_diagram.md       # Mermaid system diagram (this architecture)
│   └── data_flow.md            # Original scoring pipeline diagram
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI runner, 6 profiles, logging, output
│   └── recommender.py          # Core logic: loader, embedder, scorer, ranker
├── tests/
│   └── test_recommender.py     # 7 tests — fallback path + semantic reliability
├── model_card.md               # Detailed bias and evaluation analysis
├── reflection.md               # Profile comparison notes
├── pytest.ini                  # Sets pythonpath = . for test imports
└── requirements.txt            # Pinned dependencies
```

---

## Additional Documentation

- [Model Card](model_card.md) — full bias analysis, evaluation methodology, and per-profile findings
- [System Diagram](docs/system_diagram.md) — Mermaid flowchart of all components and data paths
- [Reflection](reflection.md) — profile comparison notes