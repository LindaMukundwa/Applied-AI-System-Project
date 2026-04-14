import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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

class Recommender:
    """OOP wrapper around the scoring recipe; used by the test suite."""

    def __init__(self, songs: List[Song]):
        """Stores the song catalog for repeated recommendation calls."""
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Returns (score, reasons) for one song: +2.0 genre, +1.0 mood, 0–1 energy similarity."""
        score = 0.0
        reasons = []

        if song.genre == user.favorite_genre:
            score += 2.0
            reasons.append(f"genre match ({song.genre}) +2.0")

        if song.mood == user.preferred_mood:
            score += 1.0
            reasons.append(f"mood match ({song.mood}) +1.0")

        energy_sim = round(1.0 - abs(song.energy - user.target_energy), 2)
        score += energy_sim
        reasons.append(f"energy similarity {energy_sim:.2f}")

        return round(score, 4), reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Scores every song in the catalog and returns the top k sorted by score."""
        scored = [(song, self._score(user, song)[0]) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a semicolon-joined string of scoring reasons for one song."""
        _, reasons = self._score(user, song)
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Reads data/songs.csv and returns a list of song dicts with typed numeric fields."""
    songs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            songs.append({
                'id':            int(row['id']),
                'title':         row['title'],
                'artist':        row['artist'],
                'genre':         row['genre'],
                'mood':          row['mood'],
                'energy':        float(row['energy']),
                'tempo_bpm':     float(row['tempo_bpm']),
                'valence':       float(row['valence']),
                'danceability':  float(row['danceability']),
                'acousticness':  float(row['acousticness']),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Scores one song against user preferences; returns (score, reasons) with max 4.0."""
    score = 0.0
    reasons = []

    if song['genre'] == user_prefs['favorite_genre']:
        score += 2.0
        reasons.append(f"genre match ({song['genre']}) +2.0")

    if song['mood'] == user_prefs['preferred_mood']:
        score += 1.0
        reasons.append(f"mood match ({song['mood']}) +1.0")

    energy_sim = round(1.0 - abs(song['energy'] - user_prefs['target_energy']), 2)
    score += energy_sim
    reasons.append(f"energy similarity {energy_sim:.2f} (target {user_prefs['target_energy']}, song {song['energy']})")

    return round(score, 4), reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Scores every song, sorts by score descending, and returns the top k as (song, score, explanation) tuples."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, "; ".join(reasons)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
