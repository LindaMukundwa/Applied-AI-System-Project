# Data Flow Diagram

This diagram traces how a single song moves from `data/songs.csv` through the
scoring pipeline to a ranked recommendation list.

Render this diagram in any Mermaid-compatible viewer — VS Code preview,
the [Mermaid Live Editor](https://mermaid.live), or GitHub (which renders
Mermaid fences natively in Markdown files).

```mermaid
flowchart TD
    A["User Preferences\nfavorite_genre · preferred_mood · target_energy"] --> R
    B["data/songs.csv"] --> C["load_songs()\ncsv.DictReader → List of song dicts"]
    C --> R["recommend_songs(user_prefs, songs, k=5)"]
    R --> G

    subgraph LOOP["Loop, score_song() called once per song"]
        G["Genre match?\nsong.genre == user.favorite_genre"]
        G -->|"Yes"| H["+2.0 pts"]
        G -->|"No"| I["+0.0 pts"]
        H --> J["Mood match?\nsong.mood == user.preferred_mood"]
        I --> J
        J -->|"Yes"| K["+1.0 pts"]
        J -->|"No"| L["+0.0 pts"]
        K --> N["Energy similarity\n1 − |song.energy − target_energy|\nadds 0.0 – 1.0 pts"]
        L --> N
        N --> O["return (score, reasons)"]
    end

    O --> P["Collect all 20\n(song, score, explanation) tuples"]
    P --> Q["Sort descending by score"]
    Q --> S["Slice top K\n→ [(song, score, explanation), ...]"]
    S --> T["main.py prints:\nTitle — Score: X.XX\nBecause: genre match · mood match · energy sim"]
```
