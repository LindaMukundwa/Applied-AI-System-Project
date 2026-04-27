import csv
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EMBEDDER_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Song:
    """Represents a single song and its audio attributes loaded from the catalog."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Holds a listener's taste preferences used as targets in the scoring recipe."""
    favorite_genre: str
    preferred_mood: str
    target_energy: float
    target_acousticness: float
    target_valence: float
    target_tempo: float


def build_embedder() -> Optional[Any]:
    """Load the sentence-transformers model; returns None and logs a warning on failure."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model '%s'...", EMBEDDER_MODEL)
        model = SentenceTransformer(EMBEDDER_MODEL)
        logger.info("Model loaded successfully.")
        return model
    except Exception as exc:
        logger.warning(
            "Could not load sentence-transformers (%s). "
            "Falling back to exact-match scoring.",
            exc,
        )
        return None


def embed_catalog(songs: List[Dict], embedder: Any) -> List[Dict]:
    """Pre-compute genre and mood embeddings for every song in the catalog."""
    genres = [s["genre"] for s in songs]
    moods = [s["mood"] for s in songs]

    logger.info("Computing embeddings for %d songs...", len(songs))
    genre_embs = embedder.encode(genres, convert_to_numpy=True, show_progress_bar=False)
    mood_embs = embedder.encode(moods, convert_to_numpy=True, show_progress_bar=False)

    for song, g_emb, m_emb in zip(songs, genre_embs, mood_embs):
        song["_genre_emb"] = g_emb
        song["_mood_emb"] = m_emb

    logger.info("Catalog embeddings ready. Semantic scoring is active.")
    return songs


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [0, 1] between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def _clamp_energy(value: float, label: str) -> float:
    """Clamp energy to [0, 1] and warn if the raw value was out of range."""
    if not (0.0 <= value <= 1.0):
        logger.warning("Energy value %.3f for '%s' is out of [0, 1]; clamping.", value, label)
        return max(0.0, min(1.0, value))
    return value


# ---------------------------------------------------------------------------
# OOP interface (used by the test suite)
# ---------------------------------------------------------------------------

class Recommender:
    """OOP wrapper around the scoring recipe; used by the test suite."""

    def __init__(self, songs: List[Song], embedder: Optional[Any] = None):
        """Stores the song catalog and optionally pre-embeds it for semantic scoring."""
        self.songs = songs
        self._embedder = embedder
        self._song_embeddings: Dict[int, Dict[str, np.ndarray]] = {}

        if embedder is not None:
            genres = [s.genre for s in songs]
            moods = [s.mood for s in songs]
            g_embs = embedder.encode(genres, convert_to_numpy=True, show_progress_bar=False)
            m_embs = embedder.encode(moods, convert_to_numpy=True, show_progress_bar=False)
            for song, g, m in zip(songs, g_embs, m_embs):
                self._song_embeddings[song.id] = {"genre": g, "mood": m}

    def _score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Return (score, reasons) using semantic similarity when embeddings are available."""
        score = 0.0
        reasons = []

        song_embs = self._song_embeddings.get(song.id)
        if song_embs is not None and self._embedder is not None:
            user_genre_emb = self._embedder.encode(user.favorite_genre, convert_to_numpy=True)
            user_mood_emb = self._embedder.encode(user.preferred_mood, convert_to_numpy=True)

            genre_sim = _cosine_sim(song_embs["genre"], user_genre_emb)
            genre_pts = round(genre_sim * 2.0, 4)
            score += genre_pts
            reasons.append(
                f"genre similarity {genre_sim:.2f} ({song.genre} \u2192 {user.favorite_genre}) +{genre_pts:.2f}"
            )

            mood_sim = _cosine_sim(song_embs["mood"], user_mood_emb)
            mood_pts = round(mood_sim * 1.0, 4)
            score += mood_pts
            reasons.append(
                f"mood similarity {mood_sim:.2f} ({song.mood} \u2192 {user.preferred_mood}) +{mood_pts:.2f}"
            )
        else:
            if song.genre == user.favorite_genre:
                score += 2.0
                reasons.append(f"genre match ({song.genre}) +2.0")
            if song.mood == user.preferred_mood:
                score += 1.0
                reasons.append(f"mood match ({song.mood}) +1.0")

        song_energy = _clamp_energy(song.energy, song.title)
        energy_sim = round(1.0 - abs(song_energy - user.target_energy), 2)
        score += energy_sim
        reasons.append(f"energy similarity {energy_sim:.2f}")

        return round(score, 4), reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Score every song and return the top k sorted by score descending."""
        scored = [(song, self._score(user, song)[0]) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a semicolon-joined string of scoring reasons for one song."""
        _, reasons = self._score(user, song)
        return "; ".join(reasons)


# ---------------------------------------------------------------------------
# Functional interface (used by main.py)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Read data/songs.csv and return a list of song dicts with typed numeric fields."""
    songs = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                songs.append({
                    "id":           int(row["id"]),
                    "title":        row["title"],
                    "artist":       row["artist"],
                    "genre":        row["genre"],
                    "mood":         row["mood"],
                    "energy":       float(row["energy"]),
                    "tempo_bpm":    float(row["tempo_bpm"]),
                    "valence":      float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                })
    except FileNotFoundError:
        logger.error("Catalog file not found: %s", csv_path)
        raise
    except (KeyError, ValueError) as exc:
        logger.error("Malformed catalog row: %s", exc)
        raise

    logger.info("Loaded %d songs from '%s'.", len(songs), csv_path)
    return songs


def score_song(
    user_prefs: Dict,
    song: Dict,
    user_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[float, List[str]]:
    """Score one song against user preferences; returns (score, reasons) with max ~4.0.

    When user_embeddings and song embeddings are present, genre and mood similarity
    are computed via cosine distance instead of exact string equality.
    """
    score = 0.0
    reasons = []

    if (
        user_embeddings is not None
        and "_genre_emb" in song
        and "_mood_emb" in song
    ):
        genre_sim = _cosine_sim(song["_genre_emb"], user_embeddings["genre"])
        genre_pts = round(genre_sim * 2.0, 4)
        score += genre_pts
        reasons.append(
            f"genre similarity {genre_sim:.2f} "
            f"({song['genre']} \u2192 {user_prefs['favorite_genre']}) +{genre_pts:.2f}"
        )

        mood_sim = _cosine_sim(song["_mood_emb"], user_embeddings["mood"])
        mood_pts = round(mood_sim * 1.0, 4)
        score += mood_pts
        reasons.append(
            f"mood similarity {mood_sim:.2f} "
            f"({song['mood']} \u2192 {user_prefs['preferred_mood']}) +{mood_pts:.2f}"
        )
    else:
        if song["genre"] == user_prefs["favorite_genre"]:
            score += 2.0
            reasons.append(f"genre match ({song['genre']}) +2.0")

        if song["mood"] == user_prefs["preferred_mood"]:
            score += 1.0
            reasons.append(f"mood match ({song['mood']}) +1.0")

    song_energy = _clamp_energy(song["energy"], song["title"])
    energy_sim = round(1.0 - abs(song_energy - user_prefs["target_energy"]), 2)
    score += energy_sim
    reasons.append(
        f"energy similarity {energy_sim:.2f} "
        f"(target {user_prefs['target_energy']}, song {song['energy']})"
    )

    return round(score, 4), reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    embedder: Optional[Any] = None,
) -> List[Tuple[Dict, float, str]]:
    """Score every song, sort descending, return top k as (song, score, explanation) tuples.

    Passes pre-computed user embeddings into score_song so the embedding model
    is called once per recommendation request, not once per song.
    """
    user_embeddings = None
    if embedder is not None:
        logger.debug("Embedding user profile for '%s'/'%s'.",
                     user_prefs["favorite_genre"], user_prefs["preferred_mood"])
        genre_emb = embedder.encode(user_prefs["favorite_genre"], convert_to_numpy=True)
        mood_emb = embedder.encode(user_prefs["preferred_mood"], convert_to_numpy=True)
        user_embeddings = {"genre": genre_emb, "mood": mood_emb}

    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, user_embeddings)
        scored.append((song, score, "; ".join(reasons)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
