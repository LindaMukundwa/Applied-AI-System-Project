# System Diagram

Render in any Mermaid-compatible viewer, VS Code preview, the [Mermaid Live Editor](https://mermaid.live), or GitHub (which renders Mermaid fences natively).

```mermaid
flowchart TD
    %% ── INPUTS ────────────────────────────────────────────────────────────────
    CSV["📄 data/songs.csv\n20 songs · 10 fields each"]
    USER["👤 User Profile\nfavorite_genre · preferred_mood\ntarget_energy · (+ 3 unused fields)"]

    %% ── COMPONENT 1 · CATALOG LOADER ──────────────────────────────────────────
    subgraph LOADER["① Catalog Loader  —  load_songs()"]
        LS["Parse CSV with csv.DictReader\nCast numeric fields to float\nReturn List[Dict]"]
    end

    %% ── COMPONENT 2 · SEMANTIC EMBEDDER ───────────────────────────────────────
    subgraph EMBEDDER["② Semantic Embedder  —  build_embedder() + embed_catalog()"]
        BE["Load all-MiniLM-L6-v2\nvia sentence-transformers\nRuns once at startup"]
        EC["Encode genre + mood strings\nfor all 20 songs → numpy vectors\nAttach _genre_emb · _mood_emb to each dict"]
        BE --> EC
    end

    %% ── COMPONENT 3 · SEMANTIC SCORER ─────────────────────────────────────────
    subgraph SCORER["③ Semantic Scorer  —  score_song()"]
        UE["Embed user's genre + mood\nonce per query"]
        SS["cosine_sim(song._genre_emb, user_genre_emb) × 2.0\ncosine_sim(song._mood_emb,  user_mood_emb)  × 1.0\n1 − |song.energy − target_energy|          × 1.0\n─────────────────────────────────────────────\nTotal score  0 – 4.0  +  explanation list"]
        UE --> SS
    end

    %% ── COMPONENT 4 · RANKER ───────────────────────────────────────────────────
    subgraph RANKER["④ Ranker  —  recommend_songs()"]
        RS["Score all 20 songs\nSort descending\nSlice top-k\nReturn (song, score, explanation) tuples"]
    end

    %% ── OUTPUT ────────────────────────────────────────────────────────────────
    OUT["🎵 Ranked Top-5\nTitle · Artist · Genre · Mood · Energy\nScore X.XX / 4.00\nReason breakdown per line"]

    %% ── HUMAN-IN-THE-LOOP ─────────────────────────────────────────────────────
    subgraph HUMAN_LOOP["⑤ Human-in-the-Loop Evaluation"]
        H1["Human reads explanations\nChecks adversarial profiles\nLooks for surprising rankings"]
        H2["Identifies weaknesses:\n• ghost genre ceiling\n• energy-mood conflict\n• ignored dimensions"]
    end

    %% ── TEST SUITE ─────────────────────────────────────────────────────────────
    subgraph TESTING["⑥ Automated Tests  —  pytest tests/"]
        T1["Recommender class\n(exact-match fallback path)\n\ntest_recommend_returns_songs_sorted_by_score\ntest_explain_recommendation_returns_non_empty_string"]
        T2["✅ 2 / 2 pass"]
        T1 --> T2
    end

    %% ── FALLBACK PATH ──────────────────────────────────────────────────────────
    FALLBACK["⚠️  Fallback\nIf model unavailable:\nlog WARNING\nuse exact-match == scoring\nApp still runs"]

    %% ── DATA FLOW ──────────────────────────────────────────────────────────────
    CSV       --> LS
    LS        --> BE
    EC        --> SS
    USER      --> UE
    SS        --> RS
    RS        --> OUT
    OUT       --> H1
    H1        --> H2

    LS        -. "Song dicts\n(no embeddings)" .-> T1

    BE        -. "Import fails?\nlog + return None" .-> FALLBACK
    FALLBACK  -. "score_song falls back\nto == operator" .-> SS
```

## Component Summary

| # | Component | Function | Key file |
|---|---|---|---|
| ① | Catalog Loader | Parses CSV → typed dicts | `src/recommender.py · load_songs()` |
| ② | Semantic Embedder | Loads model, pre-computes song vectors | `src/recommender.py · build_embedder(), embed_catalog()` |
| ③ | Semantic Scorer | Cosine similarity replaces `==` for genre/mood | `src/recommender.py · score_song()` |
| ④ | Ranker | Sorts scored songs, returns top-k with explanations | `src/recommender.py · recommend_songs()` |
| ⑤ | Human-in-the-loop | Reads output, judges quality, identifies failure modes | `src/main.py` prints; human inspects |
| ⑥ | Test Suite | Validates sort order and explanation format via fallback path | `tests/test_recommender.py` |
