# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Real-world music recommenders like Spotify and YouTube combine two main strategies: collaborative filtering, which finds patterns across millions of users to surface songs that people with similar taste enjoyed, and content-based filtering, which compares the audio attributes of songs a user already likes to find others that sound and feel similar. At scale, these platforms also layer in contextual signals like time of day, device, and activity to fine-tune what gets surfaced. This simulation focuses on content-based filtering only. It skips genre labels entirely because genre is a curatorial tag, not a felt experience and instead prioritizes the features that most directly shape how a song sounds and feels: energy level, acoustic texture, and emotional tone. The result is a scoring system that rewards closeness to a user's preferences rather than rewarding any single attribute being high or low, and a ranking step that sorts the full catalog by fit score to produce an ordered recommendation list.

---

## How The System Works

### `Song` Features Used in Scoring

Each song carries ten attributes from `data/songs.csv`. The current recipe uses three of them directly; the rest are stored on the object for potential future use.

| Feature | Type | Role in scoring |
|---|---|---|
| `genre` | string | Discrete bonus — **+2.0 pts** if it matches the user's `favorite_genre` |
| `mood` | string | Discrete bonus — **+1.0 pt** if it matches the user's `preferred_mood` |
| `energy` | float 0–1 | Continuous similarity — `1 − \|song.energy − target_energy\|`, adds 0.0–1.0 pts |
| `acousticness` | float 0–1 | Stored, not scored in current recipe |
| `valence` | float 0–1 | Stored, not scored in current recipe |
| `tempo_bpm` | float | Stored, not scored in current recipe |
| `danceability` | float 0–1 | Stored, not scored in current recipe |

### `UserProfile` Fields

| Field | Type | What it represents |
|---|---|---|
| `favorite_genre` | string | Matched against `song.genre` — triggers the +2.0 genre bonus |
| `preferred_mood` | string | Matched against `song.mood` — triggers the +1.0 mood bonus |
| `target_energy` | float 0–1 | Preferred energy level used in the similarity calculation |
| `target_acousticness` | float 0–1 | Stored, not used in current recipe |
| `target_valence` | float 0–1 | Stored, not used in current recipe |
| `target_tempo` | float (normalized) | Stored, not used in current recipe |

### Algorithm Recipe

```
score = 0.0

if song.genre == user.favorite_genre:
    score += 2.0                                   # genre bonus (discrete)

if song.mood == user.preferred_mood:
    score += 1.0                                   # mood bonus (discrete)

score += 1 - |song.energy - user.target_energy|   # energy similarity (continuous, 0–1)
```

**Max possible score: 4.0** (genre match + mood match + perfect energy alignment).

The 2:1 genre-to-mood ratio reflects a deliberate design choice: genre defines the
listening context (lofi vs. metal are completely different environments), while mood is
a softer, more fluid preference — a "chill" listener might still enjoy a "focused" track
within the same genre.

See [docs/data_flow.md](docs/data_flow.md) for a Mermaid diagram that visualizes the
full pipeline from CSV to ranked output.

### How Songs Are Ranked

All songs in the catalog are scored. The ranking rule sorts them by score descending
and returns the top `k` as `(song, score, explanation)` tuples.

### Potential Biases

- **Genre over-prioritization.** Genre is worth +2.0 — twice the mood bonus and
  potentially more than the entire energy similarity range (0–1). A song that perfectly
  matches the user's energy and mood but differs in genre will be outranked by a weak
  genre match with poor energy fit. This means users may miss great cross-genre songs.

- **Catalog representation bias.** The 20-song catalog is unevenly distributed across
  genres: lofi has 3 entries while blues, gospel, and country each have only 1. Users
  whose preferred genre is under-represented have fewer candidates to score highly,
  making top results feel repetitive.

- **Static taste profile.** The system has no memory. It cannot learn from listening
  history or adapt when the user's mood changes. Every session treats the user as if
  their preferences are fixed, which real recommenders avoid by blending in recent
  play signals.

---

### Terminal Image Output
<a><img src="terminal_output.png" alt="Terminal Output" width="800"/></a>

### Stress Test with Diverse Profiles Terminal Ouput
<a><img src="profile_recommendations.png" alt="Terminal Output" width="800"/></a>

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

**Six listener profiles, three normal and three adversarial.**
The system was tested against six distinct user preference dictionaries. The first
three were realistic listener types — a daytime pop fan, a lofi study listener, and a
rock headbanger. The final three were designed to stress-test the scoring logic:

- **Energy-Mood Conflict:** A user who asked for blues/sad music but set a very high
  energy target (0.90). The only blues/sad song in the catalog is slow and quiet
  (energy 0.38). The genre and mood bonuses still pushed it to #1 with a score of 3.48,
  ahead of genuinely high-energy songs that scored under 1.0. The system was logically
  correct but produced a result no real listener would want.

- **Ghost Genre:** A user whose favorite genre (k-pop) does not exist in the catalog.
  No song ever earned the genre bonus. The listener's maximum reachable score was 2.0
  out of 4.0 — half the ceiling available to a pop or lofi listener.

- **Ignored Dimensions:** A user who specified precise acousticness, valence, and tempo
  preferences alongside a classical/peaceful profile. Autumn Sonata scored a perfect
  4.0 — but the system never consulted any of the three extra fields. The high score
  created a false impression of accuracy.

**Weight shift experiment.**
Genre weight was halved from +2.0 to +1.0 and energy was doubled from 0–1.0 to
0–2.0, keeping the maximum score at 4.0. The most meaningful result: for the pop
profile, Gym Hero (pop/intense) dropped from #2 to #4. Songs with the right mood but
a different genre now had enough energy weight to outrank it, which felt more accurate.
However, the adversarial energy-mood conflict case was only partially improved — Last
Train South still ranked first, just with a narrower margin. The original weights were
restored after the experiment.

---

## Limitations and Risks

- **The genre bonus is too powerful for a 20-song catalog.** Genre is worth +2.0 —
  half the total score — but most genres have only one song. That single song wins its
  genre bonus automatically with no competition. This makes the genre bonus behave like
  a hard filter rather than a scoring factor.

- **Three user preferences are collected but never scored.** Acousticness, valence,
  and tempo are stored in the user profile but the scoring recipe ignores all three.
  Users who set those fields get no benefit from doing so.

- **Energy can never penalize a song.** The energy similarity formula always produces
  a positive number (minimum ~0.03). There is no floor of zero for a bad energy match,
  which allows songs with the wrong vibe to slip into results purely on proximity.

- **The catalog encodes a stereotype.** Every low-energy song has a calm or sad mood.
  Every high-energy song has an intense or happy mood. A listener who wants to run to
  sad music, or study to something upbeat, cannot be served — not because of a code
  bug, but because that combination does not exist in the data.

- **No feedback loop.** The system treats every session identically. It cannot learn
  that a user skipped the #1 result or replayed #3. Real recommenders use that signal
  constantly.

See [model_card.md](model_card.md) for a full breakdown of each bias with specific
examples and scores.

---

## Reflection

[**Full Model Card**](model_card.md) | [**Profile Comparisons**](reflection.md)

Building this recommender made clear that a system can follow its rules perfectly and
still produce results that feel wrong. The scoring formula is logically consistent
throughout but when a sad blues song ranks first for a user who asked for high-energy
music, or when an intense pop song beats a happy indie pop song, the formula is
answering the right question in the wrong context. The rules were designed for the
catalog that exists, not for the full range of human musical taste.

The deeper lesson was about where bias lives. Every low-energy song in the catalog has
a calm or sad mood, and every high-energy song has an intense or happy mood. That
assumption is baked into the data labels, not into the code. AI tools like Claude were
helpful in understanding how these decisions can impact the recommendation system as well
as providing helpful testing as well.

Adjusting the weights helped at the margins reducing the genre bonus made the mood signal 
more competitive but no weight adjustment could recommend a high-energy sad song that simply 
is not in the catalog. The most important biases in this system are not in the algorithm. They
are in the choices made about what data to include and how to label it, and those are
much harder to see and fix than a number in a formula.