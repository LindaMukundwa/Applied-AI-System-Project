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

Each song carries ten attributes from `data/songs.csv`, but only five are used in the score. The other two (`genre`, `danceability`) are stored on the object but carry zero weight, because genre labels are too coarse to reflect felt similarity and danceability is largely redundant with energy.

| Feature | Type | Role in scoring |
|---|---|---|
| `energy` | float 0–1 | Primary signal — weight 0.35 |
| `acousticness` | float 0–1 | Primary signal — weight 0.30 |
| `mood` | string | Categorical match — weight 0.20 |
| `valence` | float 0–1 | Emotional tone support — weight 0.10 |
| `tempo_bpm` | float (normalized) | Tiebreaker — weight 0.05 |
| `genre` | string | Stored, not scored — weight 0.00 |
| `danceability` | float 0–1 | Stored, not scored — weight 0.00 |

### `UserProfile` Fields

The user profile stores a preference value for each scored feature — not the user's favorite songs, but the target values the scoring rule measures closeness against.

| Field | Type | What it represents |
|---|---|---|
| `target_energy` | float 0–1 | Preferred energy level (e.g. 0.35 for chill) |
| `target_acousticness` | float 0–1 | Preferred acoustic texture (e.g. 0.80 for raw/unplugged) |
| `preferred_mood` | string | Mood label to match (e.g. `"chill"`) |
| `target_valence` | float 0–1 | Preferred emotional positivity (e.g. 0.60) |
| `target_tempo` | float (normalized) | Preferred tempo, normalized to 0–1 |
| `favorite_genre` | string | Stored for display, not used in scoring |

### How a Score Is Computed

For each numeric feature, the score is `1 - |user_target - song_value|`, which gives 1.0 for a perfect match and approaches 0.0 as the gap widens. Mood uses a binary match (1 if equal, 0 if not). The five feature scores are then multiplied by their weights and summed into a single float between 0 and 1.

### How Songs Are Ranked

All songs in the catalog are scored against the user profile. The ranking rule sorts them by score descending and returns the top `k`. Songs with equal scores are broken by higher valence.

---

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

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"