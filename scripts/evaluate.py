"""
Evaluation harness for the Music Recommender.

Runs the system on a fixed set of predefined inputs and prints a structured
pass/fail summary with confidence scores.  Requires the project root on the
Python path, which happens automatically when run as:

    python -m scripts.evaluate

Two reliability methods are exercised here:
  1. Automated assertions  — each test case states an explicit, checkable claim.
  2. Confidence scoring    — average top-1 score / 4.0 across all profiles
                             measures how strongly the system commits to its
                             recommendations (higher = more decisive).
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Callable, List

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s",
                    stream=sys.stdout)

from src.recommender import build_embedder, embed_catalog, load_genre_docs, load_songs, recommend_songs

EXACT_MATCH_GENRE_CEILING = 2.0   # max score for a ghost-genre user under old == logic


# ---------------------------------------------------------------------------
# Test-case structure
# ---------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    description: str
    profile: dict
    check: Callable[[list], bool]
    failure_hint: str
    expected_label: str = ""


@dataclass
class Result:
    case: Case
    passed: bool
    top_score: float
    top_genre: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Predefined test cases
# ---------------------------------------------------------------------------

def _make_cases() -> List[Case]:
    return [
        Case(
            name="pop_top_result_is_pop",
            description="A pop/happy/0.80 user's #1 result must be a pop song.",
            profile={"favorite_genre": "pop", "preferred_mood": "happy",
                     "target_energy": 0.80, "target_acousticness": 0.15,
                     "target_valence": 0.85, "target_tempo": 0.57},
            check=lambda recs: recs[0][0]["genre"] == "pop",
            failure_hint="Sort order or genre-match scoring is broken.",
            expected_label="top genre == pop",
        ),
        Case(
            name="pop_top_score_near_perfect",
            description="A perfect pop/happy/0.80 match should score ≥ 3.90.",
            profile={"favorite_genre": "pop", "preferred_mood": "happy",
                     "target_energy": 0.80, "target_acousticness": 0.15,
                     "target_valence": 0.85, "target_tempo": 0.57},
            check=lambda recs: recs[0][1] >= 3.90,
            failure_hint="Perfect-match score dropped below 3.90 — weight or similarity bug.",
            expected_label="top score ≥ 3.90",
        ),
        Case(
            name="lofi_top_two_are_lofi",
            description="A lofi/chill/0.38 user's top 2 results must both be lofi songs.",
            profile={"favorite_genre": "lofi", "preferred_mood": "chill",
                     "target_energy": 0.38, "target_acousticness": 0.80,
                     "target_valence": 0.58, "target_tempo": 0.20},
            check=lambda recs: recs[0][0]["genre"] == "lofi" and recs[1][0]["genre"] == "lofi",
            failure_hint="Lofi songs lost their top-2 positions — possible weight regression.",
            expected_label="top 2 genres == lofi",
        ),
        Case(
            name="rock_perfect_score",
            description="A rock/intense/0.91 user has an exact catalog match and should score 4.00.",
            profile={"favorite_genre": "rock", "preferred_mood": "intense",
                     "target_energy": 0.91, "target_acousticness": 0.08,
                     "target_valence": 0.45, "target_tempo": 0.86},
            check=lambda recs: recs[0][1] >= 3.99,
            failure_hint="Storm Runner (rock/intense/0.91) should score 4.00.",
            expected_label="top score ≥ 3.99",
        ),
        Case(
            name="ghost_genre_beats_exact_match_ceiling",
            description=(
                "A k-pop user's top score must exceed the old exact-match ceiling "
                f"of {EXACT_MATCH_GENRE_CEILING:.1f}. This proves semantic scoring "
                "is active and the description documents are improving proximity."
            ),
            profile={"favorite_genre": "k-pop", "preferred_mood": "happy",
                     "target_energy": 0.75, "target_acousticness": 0.20,
                     "target_valence": 0.82, "target_tempo": 0.50},
            check=lambda recs: recs[0][1] > EXACT_MATCH_GENRE_CEILING,
            failure_hint=(
                "k-pop top score did not exceed 2.0 — either semantic scoring is "
                "inactive or description documents are not loading."
            ),
            expected_label=f"top score > {EXACT_MATCH_GENRE_CEILING:.1f}",
        ),
        Case(
            name="sort_order_descending",
            description="All returned scores must be non-increasing (highest first).",
            profile={"favorite_genre": "electronic", "preferred_mood": "intense",
                     "target_energy": 0.90, "target_acousticness": 0.05,
                     "target_valence": 0.60, "target_tempo": 0.75},
            check=lambda recs: all(recs[i][1] >= recs[i + 1][1] for i in range(len(recs) - 1)),
            failure_hint="Ranker returned results out of order.",
            expected_label="scores non-increasing",
        ),
        Case(
            name="all_scores_in_valid_range",
            description="Every score must be in [0.0, 4.1] (slight tolerance for rounding).",
            profile={"favorite_genre": "classical", "preferred_mood": "peaceful",
                     "target_energy": 0.22, "target_acousticness": 0.99,
                     "target_valence": 0.70, "target_tempo": 0.00},
            check=lambda recs: all(0.0 <= r[1] <= 4.1 for r in recs),
            failure_hint="A score fell outside [0.0, 4.1] — clamping or weight bug.",
            expected_label="all scores in [0.0, 4.1]",
        ),
        Case(
            name="explanation_strings_non_empty",
            description="Every result must carry a non-empty explanation string.",
            profile={"favorite_genre": "jazz", "preferred_mood": "relaxed",
                     "target_energy": 0.38, "target_acousticness": 0.80,
                     "target_valence": 0.65, "target_tempo": 0.30},
            check=lambda recs: all(r[2].strip() != "" for r in recs),
            failure_hint="At least one result had an empty explanation string.",
            expected_label="all explanations non-empty",
        ),
        Case(
            name="returns_exactly_k_results",
            description="recommend_songs(k=5) must return exactly 5 results.",
            profile={"favorite_genre": "hip-hop", "preferred_mood": "focused",
                     "target_energy": 0.78, "target_acousticness": 0.10,
                     "target_valence": 0.55, "target_tempo": 0.35},
            check=lambda recs: len(recs) == 5,
            failure_hint="Returned wrong number of results.",
            expected_label="len(results) == 5",
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    songs = load_songs("data/songs.csv")
    embedder = build_embedder()
    docs: dict = {}

    if embedder is not None:
        docs = load_genre_docs("data/genre_descriptions.json", "data/mood_descriptions.json")
        embed_catalog(songs, embedder,
                      genre_docs=docs["genre"],
                      mood_docs=docs["mood"])
    else:
        print("WARNING  Semantic scoring unavailable — running in exact-match fallback mode.")
        print("         The ghost_genre_beats_exact_match_ceiling test will FAIL in this mode.")
        print()

    cases = _make_cases()
    results: List[Result] = []
    genre_docs = docs.get("genre", {}) if embedder is not None else {}
    mood_docs = docs.get("mood", {}) if embedder is not None else {}

    for case in cases:
        recs = recommend_songs(case.profile, songs, k=5, embedder=embedder,
                               genre_docs=genre_docs, mood_docs=mood_docs)
        passed = case.check(recs)
        top_score = recs[0][1] if recs else 0.0
        top_genre = recs[0][0]["genre"] if recs else "n/a"
        detail = f"top score={top_score:.2f}, top genre={top_genre}"
        results.append(Result(case=case, passed=passed,
                               top_score=top_score, top_genre=top_genre, detail=detail))

    # ── Summary ──────────────────────────────────────────────────────────────
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    avg_confidence = sum(r.top_score for r in results) / (total * 4.0)

    print()
    print("=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Tests passed    : {passed_count} / {total}")
    print(f"  Avg confidence  : {avg_confidence:.2f}  (avg top-1 score / 4.0)")

    # Ghost-genre improvement callout
    ghost = next((r for r in results if r.case.name == "ghost_genre_beats_exact_match_ceiling"), None)
    if ghost:
        arrow = "✓ IMPROVED" if ghost.passed else "✗ NO IMPROVEMENT"
        print(
            f"  Ghost genre (k-pop) top score : {ghost.top_score:.2f}  "
            f"(exact-match ceiling was {EXACT_MATCH_GENRE_CEILING:.2f})  {arrow}"
        )

    print()
    print(f"  {'RESULT':<6}  {'TEST NAME':<45}  {'DETAIL'}")
    print(f"  {'-'*6}  {'-'*45}  {'-'*30}")
    for r in results:
        status = "PASS  " if r.passed else "FAIL  "
        print(f"  {status}  {r.case.name:<45}  {r.detail}")
        if not r.passed:
            print(f"          HINT: {r.case.failure_hint}")

    print()
    if passed_count == total:
        print("  All tests passed.")
    else:
        print(f"  {total - passed_count} test(s) failed — see HINT lines above.")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
