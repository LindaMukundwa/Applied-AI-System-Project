"""
Command line runner for the Music Recommender Simulation.

Runs six user profiles — three "normal" taste profiles and three
adversarial / edge-case profiles — and prints the top-5 results for each.
Profiles are designed to reveal strengths and weaknesses of the scoring recipe.

Scoring recipe (max 4.0 per song):
  +2.0  genre match        (discrete, all-or-nothing)
  +1.0  mood match         (discrete, all-or-nothing)
  +0–1  energy similarity  (continuous: 1 - |song.energy - target_energy|)

Note: target_acousticness, target_valence, and target_tempo are stored in
prefs but are NOT used by score_song — this is a known limitation that the
adversarial profiles intentionally expose.
"""

from recommender import load_songs, recommend_songs

# ---------------------------------------------------------------------------
# Catalog tempo min/max (bpm) — used to normalise target_tempo comments.
# Actual values from data/songs.csv:  min = 56 bpm,  max = 168 bpm
# normalized = (bpm - 56) / (168 - 56) = (bpm - 56) / 112
# ---------------------------------------------------------------------------

PROFILES = [
    # ── NORMAL PROFILES ──────────────────────────────────────────────────────
    {
        "label": "Profile 1 — High-Energy Pop",
        "kind":  "normal",
        "note":  "Upbeat daytime listener who wants produced, happy pop.",
        "prefs": {
            "favorite_genre":    "pop",
            "preferred_mood":    "happy",
            "target_energy":      0.80,
            "target_acousticness": 0.15,
            "target_valence":     0.85,
            "target_tempo":       0.57,   # ≈ 120 bpm
        },
    },
    {
        "label": "Profile 2 — Chill Lofi Study Session",
        "kind":  "normal",
        "note":  "Low-energy background listener studying or working quietly.",
        "prefs": {
            "favorite_genre":    "lofi",
            "preferred_mood":    "chill",
            "target_energy":      0.38,
            "target_acousticness": 0.80,
            "target_valence":     0.58,
            "target_tempo":       0.20,   # ≈ 78 bpm
        },
    },
    {
        "label": "Profile 3 — Deep Intense Rock",
        "kind":  "normal",
        "note":  "Headbanger who wants hard, driven rock at high tempo.",
        "prefs": {
            "favorite_genre":    "rock",
            "preferred_mood":    "intense",
            "target_energy":      0.91,
            "target_acousticness": 0.08,
            "target_valence":     0.45,
            "target_tempo":       0.86,   # ≈ 152 bpm
        },
    },

    # ── ADVERSARIAL / EDGE-CASE PROFILES ─────────────────────────────────────
    {
        "label": "Profile 4 — Energy-Mood Conflict  [ADVERSARIAL]",
        "kind":  "adversarial",
        "note":  (
            "A user who wants HIGH energy (0.90) but describes their mood as 'sad'. "
            "The catalog's only sad song is Last Train South (blues, energy 0.38). "
            "Watch whether genre+mood bonuses (max 3.0) overpower the energy mismatch, "
            "pushing a slow, low-energy track to #1 despite the user's energy request."
        ),
        "prefs": {
            "favorite_genre":    "blues",
            "preferred_mood":    "sad",
            "target_energy":      0.90,
            "target_acousticness": 0.20,
            "target_valence":     0.25,
            "target_tempo":       0.86,   # ≈ 152 bpm
        },
    },
    {
        "label": "Profile 5 — Ghost Genre (k-pop)  [ADVERSARIAL]",
        "kind":  "adversarial",
        "note":  (
            "A user whose favorite_genre ('k-pop') does not exist in the catalog. "
            "No song ever earns the +2.0 genre bonus, so the user's max reachable "
            "score is 2.0 (mood + perfect energy). This exposes a structural ceiling "
            "that penalises niche or unlisted-genre listeners."
        ),
        "prefs": {
            "favorite_genre":    "k-pop",   # not in catalog
            "preferred_mood":    "happy",
            "target_energy":      0.75,
            "target_acousticness": 0.20,
            "target_valence":     0.82,
            "target_tempo":       0.50,
        },
    },
    {
        "label": "Profile 6 — Ignored Dimensions (acoustic classical)  [ADVERSARIAL]",
        "kind":  "adversarial",
        "note":  (
            "A user with very specific acousticness (0.99), valence (0.70), and tempo "
            "preferences. The scoring recipe ignores all three of those fields — only "
            "genre, mood, and energy affect the score. The top result will likely score "
            "4.0 (perfect) even though the recommender never evaluated acousticness, "
            "valence, or tempo, hiding a weakness in the feature set."
        ),
        "prefs": {
            "favorite_genre":    "classical",
            "preferred_mood":    "peaceful",
            "target_energy":      0.22,   # matches Autumn Sonata exactly
            "target_acousticness": 0.99,  # ignored by score_song
            "target_valence":     0.70,   # ignored by score_song
            "target_tempo":       0.00,   # ignored by score_song (≈ 56 bpm)
        },
    },
]


def run_profile(profile: dict, songs: list) -> None:
    """Runs the recommender for one profile and prints a formatted result block."""
    print()
    print("=" * 60)
    print(f"  {profile['label']}")
    print(f"  {profile['note']}")
    print("=" * 60)

    prefs = profile["prefs"]
    print(
        f"  Genre: {prefs['favorite_genre']} | Mood: {prefs['preferred_mood']} | "
        f"Energy target: {prefs['target_energy']}"
    )
    print("-" * 60)

    recommendations = recommend_songs(prefs, songs, k=5)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")
        print(f"       Score : {score:.2f} / 4.00")
        for reason in explanation.split("; "):
            print(f"       + {reason}")


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from catalog.")
    print()
    print("Running 6 user profiles:")
    print("  Profiles 1-3  → normal taste profiles")
    print("  Profiles 4-6  → adversarial / edge-case profiles")

    for profile in PROFILES:
        run_profile(profile, songs)

    print()
    print("=" * 60)
    print("  Run complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
