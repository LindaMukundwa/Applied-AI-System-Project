"""
Command line runner for the Music Recommender Simulation.

Runs six user profiles, three "normal" taste profiles and three
adversarial / edge-case profiles, and prints the top-5 results for each.
Profiles are designed to reveal strengths and weaknesses of the scoring recipe.

Scoring recipe (max ~4.0 per song):
  Semantic mode (sentence-transformers available):
    0–2.0  genre similarity   (cosine similarity × 2.0)
    0–1.0  mood similarity    (cosine similarity × 1.0)
    0–1.0  energy similarity  (continuous: 1 - |song.energy - target_energy|)

  Fallback mode (exact-match):
    +2.0   genre match        (discrete, all-or-nothing)
    +1.0   mood match         (discrete, all-or-nothing)
    0–1.0  energy similarity  (continuous)
"""

import logging
import sys

from src.recommender import build_embedder, embed_catalog, load_songs, recommend_songs

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PROFILES = [
    # ── NORMAL PROFILES ──────────────────────────────────────────────────────
    {
        "label": "Profile 1 : High-Energy Pop",
        "kind":  "normal",
        "note":  "Upbeat daytime listener who wants produced, happy pop.",
        "prefs": {
            "favorite_genre":     "pop",
            "preferred_mood":     "happy",
            "target_energy":       0.80,
            "target_acousticness": 0.15,
            "target_valence":      0.85,
            "target_tempo":        0.57,
        },
    },
    {
        "label": "Profile 2 : Chill Lofi Study Session",
        "kind":  "normal",
        "note":  "Low-energy background listener studying or working quietly.",
        "prefs": {
            "favorite_genre":     "lofi",
            "preferred_mood":     "chill",
            "target_energy":       0.38,
            "target_acousticness": 0.80,
            "target_valence":      0.58,
            "target_tempo":        0.20,
        },
    },
    {
        "label": "Profile 3 : Deep Intense Rock",
        "kind":  "normal",
        "note":  "Headbanger who wants hard, driven rock at high tempo.",
        "prefs": {
            "favorite_genre":     "rock",
            "preferred_mood":     "intense",
            "target_energy":       0.91,
            "target_acousticness": 0.08,
            "target_valence":      0.45,
            "target_tempo":        0.86,
        },
    },

    # ── ADVERSARIAL / EDGE-CASE PROFILES ─────────────────────────────────────
    {
        "label": "Profile 4 : Energy-Mood Conflict  [ADVERSARIAL]",
        "kind":  "adversarial",
        "note":  (
            "A user who wants HIGH energy (0.90) but describes their mood as 'sad'. "
            "The catalog's only sad song is Last Train South (blues, energy 0.38). "
            "Watch whether genre+mood bonuses overpower the energy mismatch."
        ),
        "prefs": {
            "favorite_genre":     "blues",
            "preferred_mood":     "sad",
            "target_energy":       0.90,
            "target_acousticness": 0.20,
            "target_valence":      0.25,
            "target_tempo":        0.86,
        },
    },
    {
        "label": "Profile 5 : Ghost Genre (k-pop)  [ADVERSARIAL]",
        "kind":  "adversarial",
        "note":  (
            "A user whose favorite_genre ('k-pop') does not exist in the catalog. "
            "Exact-match scoring: max reachable score is 2.0 (zero genre bonus). "
            "Semantic scoring: partial credit flows from proximity to pop/electronic."
        ),
        "prefs": {
            "favorite_genre":     "k-pop",
            "preferred_mood":     "happy",
            "target_energy":       0.75,
            "target_acousticness": 0.20,
            "target_valence":      0.82,
            "target_tempo":        0.50,
        },
    },
    {
        "label": "Profile 6 : Ignored Dimensions (acoustic classical)  [ADVERSARIAL]",
        "kind":  "adversarial",
        "note":  (
            "A user with very specific acousticness, valence, and tempo preferences. "
            "Those fields do not affect the score, only genre, mood, and energy do. "
            "Semantic scoring improves genre/mood resolution but cannot fix missing fields."
        ),
        "prefs": {
            "favorite_genre":     "classical",
            "preferred_mood":     "peaceful",
            "target_energy":       0.22,
            "target_acousticness": 0.99,
            "target_valence":      0.70,
            "target_tempo":        0.00,
        },
    },
]


def run_profile(profile: dict, songs: list, embedder) -> None:
    """Run the recommender for one profile and print a formatted result block."""
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

    recommendations = recommend_songs(prefs, songs, k=5, embedder=embedder)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")
        print(f"       Score : {score:.2f} / 4.00")
        for reason in explanation.split("; "):
            print(f"       + {reason}")


def main() -> None:
    songs = load_songs("data/songs.csv")

    embedder = build_embedder()
    if embedder is not None:
        embed_catalog(songs, embedder)
        logger.info("Semantic scoring active — genre and mood use cosine similarity.")
    else:
        logger.warning("Semantic scoring unavailable — using exact-match fallback.")

    print()
    print(f"Loaded {len(songs)} songs from catalog.")
    print()
    print("Running 6 user profiles:")
    print("  Profiles 1-3  → normal taste profiles")
    print("  Profiles 4-6  → adversarial / edge-case profiles")

    for profile in PROFILES:
        run_profile(profile, songs, embedder)

    print()
    print("=" * 60)
    print("  Run complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
