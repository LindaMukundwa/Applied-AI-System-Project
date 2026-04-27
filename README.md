# Music Recommender with Semantic Scoring

The original project was a music recommender simulation from Module 4 which acted like a small music recommender system similar to applications like Spotify. It mirrored real world AI recommenders and represented songs and a user "taste profile" as data, along with other features.

### Terminal Image Output
<a><img src="/assets/Output.png" alt="Terminal Output" width="800"/></a>

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

## Testing Summary

### What the tests cover

Two automated tests in `tests/test_recommender.py` exercise the `Recommender` class (the OOP interface used directly by the test file):

- `test_recommend_returns_songs_sorted_by_score`: asserts that a pop/happy/0.80 user sees a pop, happy song at rank #1, confirming the sort order is correct.
- `test_explain_recommendation_returns_non_empty_string`: asserts that `explain_recommendation()` returns a non-empty string.

Both tests pass using the exact-match fallback path (the `Recommender` is constructed without an embedder), which confirms the fallback is functional and that the `Song` dataclass interface is intact.

### What the adversarial profiles revealed

| Profile | Finding |
|---|---|
| k-pop (ghost genre) | Semantic scoring raised the ceiling from 2.0 → 3.24/4.00 and surfaced genuinely relevant songs |
| Energy-Mood Conflict | Genre+mood bonuses still dominate; a slow blues song ranks #1 for a high-energy user, the 2:1:1 weight structure is the root cause, not the similarity function |
| Ignored Dimensions | Semantics improved genre/mood resolution but acousticness, valence, and tempo remain unscored,a perfect 4.0 is still achievable without the system ever reading those fields |

### What worked

- Replacing exact-match with cosine similarity directly solved the ghost-genre ceiling problem.
- Pre-computing catalog embeddings at startup keeps the main loop fast.
- The explanation strings (`genre similarity 0.64 (indie pop → pop) +1.27`) make partial credit auditable, a human can immediately see why a song was ranked where it was.
- The graceful fallback means the app is robust to dependency failures.

### What didn't work / known limits

- The weight structure (2:1:1) can still be gamed by genre+mood alignment even when energy is badly wrong. Semantic scoring made the genre/mood signals more accurate but did not change their relative power.
- Three user-profile fields (acousticness, valence, tempo) are still stored but never scored. The system collects information it doesn't use, which could mislead users who set those fields carefully.
- The 20-song catalog is too small for 17 genres. Most genres have exactly one song, so the genre bonus behaves like a hard filter rather than a differentiator.

### What would come next

Adding semantically-scored acousticness and valence (both are already stored as floats, but their labels could also be embedded), expanding the catalog to 5–10 songs per genre, and a diversity rule preventing the same genre from dominating a single recommendation list would address the three remaining weaknesses without changing the core architecture.

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
│   └── test_recommender.py     # 2 pytest tests (exact-match fallback path)
├── model_card.md               # Detailed bias and evaluation analysis
├── reflection.md               # Profile comparison notes
├── pytest.ini                  # Sets pythonpath = . for test imports
└── requirements.txt            # Pinned dependencies
```

---

## Additional Documentation

- [Model Card](model_card.md) : full bias analysis, evaluation methodology, and per-profile findings
- [System Diagram](docs/system_diagram.md) : Mermaid flowchart of all components and data paths
- [Reflection](reflection.md) : profile comparison notes


## Reflection

This is the final project of an entire course surrounding utilizing AI as a human-in-the-loop. As the capabilities of AI continue to grow, it cannot be understated how crucial learning and leveraging the skills it has to becoming a better engineer. This project has taught me important qualities similar to being a principal architect or senior software engineer. Having an understanding of the core project being built and focusing on getting a concrete plan is imperative and it was used heavily in the development in this project. In terms of problem solving, this is where our thinking and debugging cannot be replaced, by having a clear understanding of what the project is expected to do, conversations about tradeoffs become obsolete because mistakes are less prone to happen since you have a full grasp how it should operate. This is a project that makes me feel more comfortable and confident to continue in this indistry as we adopt more of these tools in our day-to-day workflow and I am excited to keep growing even more!