# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeMatch 1.0**

A rule-based music recommender that matches songs to a listener's preferred genre,
mood, and energy level using a simple point-based scoring recipe.

---

## 2. Intended Use

This system is built for classroom exploration only. It is not intended for a real
product.

It suggests up to five songs from a small 20-song catalog based on three things a
user tells it: their favorite genre, their preferred mood, and how high-energy they
want the music to feel.

The system assumes the user already knows what they want and can express it clearly.
It does not learn from listening history, it does not adapt over time, and it does
not know anything about a user beyond the three preferences they provide.

It is a good starting point for understanding how content-based recommenders work,
but it is not ready for real users.

---

## 3. How the Model Works

Every song in the catalog gets a score between 0 and 4. The score is built from
three pieces:

**Genre match:** If the song's genre matches the user's favorite genre, it gets 2
bonus points. If not, it gets nothing. There is no middle ground — "rock" and "metal"
are treated as completely different, even though a real listener might enjoy both.

**Mood match:** If the song's mood matches what the user asked for, it gets 1 bonus
point. Same rule — it either matches or it doesn't.

**Energy closeness:** Every song gets a score based on how close its energy level is
to the user's target. A perfect energy match scores 1.0. A song that is very far away
in energy scores close to 0, but it never scores exactly 0 — even a totally wrong
energy still adds a tiny amount.

After all three pieces are added together, the songs are sorted from highest to lowest
score and the top five are returned. Each result also shows a written explanation of
exactly which pieces contributed to its score.

---

## 4. Data

The catalog has 20 songs stored in `data/songs.csv`. Each song has ten attributes:
a title, an artist, a genre, a mood, and six numeric features (energy, tempo, valence,
danceability, and acousticness).

The catalog covers 17 different genres including pop, lofi, rock, metal, jazz, blues,
classical, and electronic. It covers 13 different moods including happy, chill,
intense, sad, melancholic, and peaceful.

The coverage is very uneven. Lofi has 3 songs, pop has 2, and every other genre has
exactly 1. Happy, chill, and intense moods each appear in 3 songs; moods like sad,
peaceful, angry, and romantic appear in only 1 song each.

No songs were added or removed from the original dataset.

The dataset reflects a mainstream, Western view of music. Genres like K-pop, Afrobeats,
or classical Indian music are not represented. A listener whose favorite genre is not
in the catalog can never earn the genre bonus — they are permanently capped at a lower
maximum score than listeners whose genre happens to be covered.

The catalog also has a noticeable energy gap: no songs have energy between 0.55 and
0.72. Listeners who prefer moderate-energy music — not quiet background music, but not
high-tempo workout music either — are the most poorly served group in the catalog.

---

## 5. Strengths

The system works best when the user's preferences match a well-represented part of
the catalog.

Lofi study listeners are served well. There are three lofi songs and three chill
songs, so the system has real variety to rank within that genre. The top two results
(Library Rain and Midnight Coding) felt genuinely appropriate for a background study
session when tested.

High-intensity rock listeners also do well. Storm Runner (rock/intense) scored a
perfect 4.0 for the Deep Intense Rock profile — genre, mood, and energy all aligned
exactly. The system correctly identified it as the best possible match.

Transparency is the biggest practical strength. Every recommendation comes with a
written explanation showing exactly why each song scored what it scored. Most real-world
recommenders give no reasons at all. Being able to read "genre match (+2.0), mood match
(+1.0), energy similarity 0.98" makes it easy to understand what the system is actually
doing and to catch when it goes wrong.

---

## 6. Limitations and Bias

**Genre dominates everything.**
The genre bonus is worth 2.0 points — half of the total possible score. This means a
genre-matching song will almost always outrank a non-genre-matching song, even when the
non-matching song fits the user's mood and energy much better. For example, a pop
listener who wants happy music receives Gym Hero (pop/intense) at #2. Gym Hero is pop,
but it is labeled "intense" — not happy. Songs that matched the happy mood but had a
different genre ranked lower, because no combination of mood and energy can reliably
beat the genre bonus. More detail on this in the Evaluation section below.

**Genres and moods are matched by exact spelling only.**
"indie pop" and "pop" score zero genre overlap. "jazz" and "blues" score zero overlap.
The system treats every genre as completely unrelated to every other genre. A jazz fan
could genuinely love a blues track, but the algorithm has no way to know that.

**Some users have a structural ceiling.**
A listener who prefers a genre not in the catalog — like K-pop or Afrobeats — can never
earn the genre bonus. Their maximum reachable score is 2.0 out of 4.0. A pop listener
can reach 4.0. This is not a bug the user caused — it is a structural unfairness baked
into the weight system.

**Energy never hurts a song.**
A song with completely the wrong energy still earns a small positive energy score (as
low as 0.03). There is no penalty for a bad energy match — only a reduced reward. This
allows songs that are wrong in every meaningful way to still appear in results purely
because their energy number happened to be close. Last Train South (blues/sad, energy
0.38) showed up at #5 for a chill lofi study listener because its energy was an exact
match. No actual lofi listener would want that song in their playlist.

**Three preference fields are silently ignored.**
The user profile stores acousticness, valence, and tempo. The scoring recipe never
reads any of them. A user who carefully specifies they want highly acoustic music gets
the same result as one who asked for fully produced electronic music — as long as
genre, mood, and energy match. This is invisible to the user, which makes it worse.

**The catalog has a stereotype baked into it.**
Every low-energy song in the catalog has a calm or sad mood. Every high-energy song
has an intense or happy mood. There is no sad high-energy song, no peaceful mid-energy
song. A listener who wants to run to sad music, or study to something upbeat, will never
find what they are looking for — not because the algorithm fails, but because the data
never included it. The bias is in the catalog, not the code, which makes it harder to
fix.

---

## 7. Evaluation

Six user profiles were tested — three realistic and three adversarial.

**Realistic profiles:**
- High-Energy Pop Listener (genre: pop, mood: happy, energy: 0.80)
- Chill Lofi Study Session (genre: lofi, mood: chill, energy: 0.38)
- Deep Intense Rock (genre: rock, mood: intense, energy: 0.91)

**Adversarial profiles (designed to find weaknesses):**
- Energy-Mood Conflict (genre: blues, mood: sad, energy: 0.90) — asking for high-energy
  music while preferring a genre and mood that only appear in one quiet, slow song.
  Result: Last Train South (energy 0.38) ranked first with a score of 3.48, beating
  every genuinely high-energy song. The genre and mood bonuses (3.0 combined) completely
  overrode the energy signal.
- Ghost Genre (genre: k-pop) — a listener whose genre does not exist in the catalog.
  No song earned the genre bonus. Maximum reachable score was 2.0 out of 4.0.
- Ignored Dimensions (genre: classical, acousticness: 0.99, valence: 0.70) — designed
  to expose the three silent fields. Autumn Sonata scored a perfect 4.0 without the
  system ever consulting acousticness, valence, or tempo.

A weight experiment was also run: genre weight halved to +1.0, energy doubled to 0–2.0,
keeping the max score at 4.0. The most meaningful result was that Gym Hero dropped from
#2 to #4 for the High-Energy Pop profile, because mood-matching songs now had enough
energy weight to outrank a genre match with the wrong mood. The adversarial energy-mood
conflict case improved (margin shrank from 2.49 to 0.98) but was not resolved — Last
Train South still won. The original weights were restored after the experiment.

The biggest surprise: a score of 3.98 out of 4.00 does not mean the recommendation is
good. It just means the song matched the numbers. The score has no opinion on whether
the result actually sounds like what the user wanted.

---

## 8. Future Work

**Score acousticness, valence, and tempo.**
These fields are already collected for every song and stored in the user profile.
Adding even a small weight to each — for example, 0.3 points for a close acousticness
match — would make the system more accurate and would finally put those fields to use.

**Give partial credit for related genres.**
Instead of all-or-nothing genre matching, similar genres could score a partial bonus.
A pop listener finding an indie pop song could score 1.0 instead of 2.0. A jazz
listener finding a blues song could score 1.0. This would reduce the genre monopoly
and surface more variety.

**Add a diversity rule to the ranking step.**
The current system always returns the five highest-scoring songs with no awareness of
variety. A simple rule — for example, no two songs from the same genre or artist in one
recommendation list — would make results feel less repetitive and give users a wider
view of the catalog.

**Expand the catalog.**
Twenty songs is not enough to serve 17 genres fairly. Adding 5 to 10 songs per genre,
and including songs in the 0.55–0.72 energy range that is currently empty, would make
the scoring differences between songs more meaningful and reduce the winner-takes-all
effect of the genre bonus.

---

## 9. Personal Reflection

Building this recommender made one thing very clear: a system can be technically correct
and still feel wrong. The scoring formula does exactly what it is supposed to do. It
follows the rules precisely and always returns results in the right order. But when a
sad blues song shows up in a chill lofi study playlist, or when a listener who loves
K-pop is permanently capped at half the possible score, the formula is giving the right
answer to the wrong question.

The most unexpected discovery was how much the data itself shapes the recommendations —
sometimes more than the algorithm does. The catalog was built assuming that low energy
always means calm and high energy always means happy or intense. That assumption is in
the mood labels, not in the code. No matter how the weights are adjusted, the system
cannot recommend a high-energy sad song because no such song exists in the catalog.
The bias lives in the data, which makes it much harder to spot and harder to fix than
a bug in a formula.

This project changed how I think about real music apps. Spotify or YouTube Music feel
smart because they have millions of songs, rich audio features for each one, and a
feedback loop that learns every time a listener skips a track or replays one. This
simulation has 20 songs, three active features, and no feedback. The gap between the
two is not mainly about the algorithm being smarter — it is about having more data and
the ability to learn from being wrong. A simple algorithm with rich, representative
data will outperform a sophisticated algorithm running on a small, biased catalog every
time.
