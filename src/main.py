"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Taste profile: "Daytime Pop Listener"
    # Models a listener who wants upbeat, high-energy, happy pop tracks.
    # tempo_bpm ~120 normalized: (120 - 56) / (168 - 56) ≈ 0.57
    #   (56 and 168 are the min/max bpm values in the current catalog)
    user_prefs = {
        "favorite_genre":    "pop",    # genre match triggers +2.0 bonus
        "preferred_mood":    "happy",  # mood match triggers +1.0 bonus
        "target_energy":      0.80,    # high energy
        "target_acousticness": 0.15,   # prefers produced sound over acoustic
        "target_valence":     0.85,    # bright, positive tone
        "target_tempo":       0.57,    # ~120 bpm, normalized
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 52)
    print(f"  Top {len(recommendations)} Recommendations")
    print(f"  Profile: {user_prefs['favorite_genre']} / {user_prefs['preferred_mood']} / energy {user_prefs['target_energy']}")
    print("=" * 52)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n#{rank}  {song['title']}  —  {song['artist']}")
        print(f"    Score : {score:.2f} / 4.00")
        for reason in explanation.split("; "):
            print(f"    + {reason}")


if __name__ == "__main__":
    main()
