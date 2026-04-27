"""
Streamlit UI for the Music Recommender Simulation.

Run locally:
    streamlit run app.py

Deploy to HuggingFace Spaces:
    Push the repo to a HF Space (Streamlit SDK).
    The model (~80 MB) downloads on first run and is cached for the session.
"""

import logging
import streamlit as st

logging.basicConfig(level=logging.WARNING)

from src.recommender import (
    build_embedder,
    embed_catalog,
    load_genre_docs,
    load_songs,
    recommend_songs,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# ── Load system once and cache ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model — this takes a few seconds on first run...")
def load_system():
    catalog = load_songs("data/songs.csv")
    embedder = build_embedder()
    docs: dict = {}
    if embedder is not None:
        docs = load_genre_docs(
            "data/genre_descriptions.json",
            "data/mood_descriptions.json",
        )
        embed_catalog(catalog, embedder,
                      genre_docs=docs.get("genre", {}),
                      mood_docs=docs.get("mood", {}))
    return catalog, embedder, docs


songs, embedder, docs = load_system()

CATALOG_GENRES = sorted({s["genre"] for s in songs})
CATALOG_MOODS  = sorted({s["mood"]  for s in songs})

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎵 Your Taste Profile")
    st.caption(
        "Set your preferences and the system will rank every song in the "
        "catalog by how well it fits."
    )
    st.divider()

    # Genre — catalog pick or free-text (demonstrates ghost-genre improvement)
    genre_options = CATALOG_GENRES + ["✏️ Type my own..."]
    selected_genre = st.selectbox(
        "Favorite genre",
        genre_options,
        index=genre_options.index("pop") if "pop" in genre_options else 0,
    )
    if selected_genre == "✏️ Type my own...":
        favorite_genre = st.text_input(
            "Enter a genre",
            value="k-pop",
            help="Try 'k-pop', 'afrobeats', or 'bossa nova' to see semantic scoring handle genres not in the catalog.",
        )
        if not favorite_genre.strip():
            favorite_genre = "k-pop"
    else:
        favorite_genre = selected_genre

    # Mood — catalog pick or free-text
    mood_options = CATALOG_MOODS + ["✏️ Type my own..."]
    selected_mood = st.selectbox(
        "Preferred mood",
        mood_options,
        index=mood_options.index("happy") if "happy" in mood_options else 0,
    )
    if selected_mood == "✏️ Type my own...":
        preferred_mood = st.text_input(
            "Enter a mood",
            value="euphoric",
        )
        if not preferred_mood.strip():
            preferred_mood = "happy"
    else:
        preferred_mood = selected_mood

    target_energy = st.slider(
        "Energy level",
        min_value=0.0, max_value=1.0, value=0.75, step=0.01,
        help="0 = quiet and calm · 1 = high-intensity",
    )

    st.divider()

    use_semantic = st.toggle(
        "Semantic scoring",
        value=True,
        disabled=embedder is None,
        help=(
            "ON  — genre and mood matched by meaning using description embeddings.\n"
            "OFF — exact string match only (original behaviour)."
        ),
    )

    k = st.slider("Number of results", min_value=1, max_value=len(songs), value=5)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🎵 Music Recommender")
st.markdown(
    "A content-based recommender that scores every song in a 20-song catalog "
    "against your taste profile. Toggle **Semantic scoring** in the sidebar to "
    "compare meaning-based matching against the original exact-match algorithm."
)

# Mode status banner
if embedder is None:
    st.error(
        "Embedding model unavailable - running in exact-match fallback mode. "
        "Install `sentence-transformers` and restart to enable semantic scoring.",
        icon="🔴",
    )
elif use_semantic:
    st.success(
        "Semantic scoring active - genre and mood matched via cosine similarity "
        "on enriched description documents.",
        icon="✅",
    )
else:
    st.warning(
        "Exact-match mode - only songs whose genre/mood labels match yours "
        "exactly earn those bonus points.",
        icon="⚠️",
    )

# ── Run recommender ────────────────────────────────────────────────────────────

user_prefs = {
    "favorite_genre":     favorite_genre,
    "preferred_mood":     preferred_mood,
    "target_energy":      target_energy,
    "target_acousticness": 0.5,
    "target_valence":      0.5,
    "target_tempo":        0.5,
}

active_embedder   = embedder          if use_semantic else None
active_genre_docs = docs.get("genre") if use_semantic else None
active_mood_docs  = docs.get("mood")  if use_semantic else None

recs = recommend_songs(
    user_prefs, songs, k=k,
    embedder=active_embedder,
    genre_docs=active_genre_docs,
    mood_docs=active_mood_docs,
)

# ── Results ────────────────────────────────────────────────────────────────────

st.subheader(
    f"Top {k} for  **{favorite_genre}** · **{preferred_mood}** · "
    f"energy **{target_energy:.2f}**"
)

for rank, (song, score, explanation) in enumerate(recs, start=1):
    with st.container(border=True):
        col_info, col_score = st.columns([4, 1])

        with col_info:
            st.markdown(f"**#{rank} — {song['title']}**  ·  *{song['artist']}*")
            st.caption(
                f"Genre: `{song['genre']}`  ·  "
                f"Mood: `{song['mood']}`  ·  "
                f"Energy: `{song['energy']:.2f}`"
            )

        with col_score:
            st.metric("Score", f"{score:.2f}", delta=f"/ 4.00",
                      delta_color="off")
            st.progress(min(score / 4.0, 1.0))

        with st.expander("Score breakdown"):
            for line in explanation.split("; "):
                st.markdown(f"- {line}")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Model: `all-MiniLM-L6-v2` via sentence-transformers  ·  "
    "Catalog: 20 songs · 17 genres  ·  "
    "Scoring: cosine similarity × weights (genre ×2, mood ×1, energy ×1)"
)
