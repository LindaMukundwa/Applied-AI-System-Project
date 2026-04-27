import math
import numpy as np
import pytest
from src.recommender import Song, UserProfile, Recommender, _cosine_sim, score_song, recommend_songs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1, title="Test Pop Track", artist="Test Artist",
            genre="pop", mood="happy", energy=0.8,
            tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2,
        ),
        Song(
            id=2, title="Chill Lofi Loop", artist="Test Artist",
            genre="lofi", mood="chill", energy=0.4,
            tempo_bpm=80, valence=0.6, danceability=0.5, acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def _song_dict(genre="pop", mood="happy", energy=0.8, genre_emb=None, mood_emb=None):
    """Build a minimal song dict, optionally with pre-attached embeddings."""
    s = {"title": "T", "artist": "A", "genre": genre, "mood": mood, "energy": energy}
    if genre_emb is not None:
        s["_genre_emb"] = genre_emb
    if mood_emb is not None:
        s["_mood_emb"] = mood_emb
    return s


# ---------------------------------------------------------------------------
# Existing tests (exact-match / OOP fallback path)
# ---------------------------------------------------------------------------

def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop", preferred_mood="happy",
        target_energy=0.8, target_acousticness=0.20,
        target_valence=0.85, target_tempo=0.57,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)
    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop", preferred_mood="happy",
        target_energy=0.8, target_acousticness=0.20,
        target_valence=0.85, target_tempo=0.57,
    )
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# Reliability tests — semantic scoring path (mock numpy vectors, no model needed)
# ---------------------------------------------------------------------------

def test_cosine_sim_identical_vectors_returns_one():
    v = np.array([1.0, 0.5, 0.0])
    assert _cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_sim_orthogonal_vectors_returns_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine_sim(a, b) == pytest.approx(0.0)


def test_score_song_perfect_embedding_match_scores_four():
    """Identical genre + mood embeddings + exact energy → score == 4.0."""
    v = np.array([1.0, 0.0])
    song = _song_dict(genre="pop", mood="happy", energy=0.8, genre_emb=v, mood_emb=v)
    user_prefs = {"favorite_genre": "pop", "preferred_mood": "happy", "target_energy": 0.8}
    user_embeddings = {"genre": v, "mood": v}
    score, reasons = score_song(user_prefs, song, user_embeddings)
    assert score == pytest.approx(4.0)
    assert len(reasons) == 3  # genre, mood, energy — one line each


def test_ghost_genre_gets_partial_credit_not_zero_not_full():
    """A song at a 45-degree embedding angle earns > 0 and < 2.0 genre points."""
    user_genre = np.array([1.0, 0.0])
    song_genre = np.array([math.sqrt(2) / 2, math.sqrt(2) / 2])  # 45° → cosine ≈ 0.707
    v_mood = np.array([1.0, 0.0])
    song = _song_dict(genre="indie pop", mood="happy", energy=0.75,
                      genre_emb=song_genre, mood_emb=v_mood)
    user_prefs = {"favorite_genre": "k-pop", "preferred_mood": "happy", "target_energy": 0.75}
    user_embeddings = {"genre": user_genre, "mood": v_mood}
    score, reasons = score_song(user_prefs, song, user_embeddings)
    # Extract genre points from the reason string: "...+1.41"
    genre_pts = float(reasons[0].split("+")[-1])
    assert 0.0 < genre_pts < 2.0


def test_recommend_songs_returns_k_results_in_descending_order():
    """recommend_songs must return exactly k results sorted by score, highest first."""
    songs = [
        _song_dict(genre="rock", mood="intense", energy=i * 0.1)
        for i in range(1, 9)
    ]
    user_prefs = {"favorite_genre": "rock", "preferred_mood": "intense", "target_energy": 0.5}
    results = recommend_songs(user_prefs, songs, k=4)
    assert len(results) == 4
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted descending by score"
