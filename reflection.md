# Reflection: Profile Comparisons and What They Revealed

This file compares the recommendation outputs across the six user profiles tested during evaluation. For each pair, I note what changed between them and why that change makes sense or why it doesn't.

---

## Pair 1: High-Energy Pop vs Chill Lofi Study Session

The High-Energy Pop profile (genre: pop, mood: happy, energy: 0.80) got Sunrise City
at #1 with a score of 3.98. The Chill Lofi profile (genre: lofi, mood: chill, energy:
0.38) got Library Rain at #1 with 3.97. Both results felt right — these were the songs
that matched all three preference fields almost perfectly.

The interesting comparison is what happened at the bottom of each list. The pop profile's
#5 result was Concrete Dreams, a hip-hop/focused track. It had no genre or mood match —
it only showed up because its energy (0.78) was close to the pop listener's target (0.80).
That is not a song a pop fan would expect to see.

The lofi profile's #5 result was even more surprising: Last Train South, a blues/sad
track, appeared because its energy (0.38) was a perfect match. A real lofi study
listener would never want a slow, melancholy blues song in their playlist — but the
algorithm could not tell the difference between "right energy, right genre" and "right
energy, completely wrong everything else."

The takeaway: both profiles were served well at the top, but the bottom of each list
reveals that energy alone is not a good enough signal to screen songs. Once genre and
mood bonuses run out, the system grabs whatever is nearest in energy, regardless of
whether it actually fits.

---

## Pair 2: Deep Intense Rock vs Energy-Mood Conflict

The Deep Intense Rock profile (genre: rock, mood: intense, energy: 0.91) got Storm
Runner at a perfect score of 4.00. Genre matched, mood matched, energy matched exactly.
This was the clearest success of all six profiles — the system did exactly what you
would want it to do.

The Energy-Mood Conflict profile was designed to break that logic. It asked for blues
genre, sad mood, but a very high energy of 0.90. The problem: the only blues song in
the catalog is Last Train South, which is quiet and slow (energy 0.38). The system
still gave it a score of 3.48 and ranked it first — way ahead of genuinely high-energy
songs like Storm Runner (0.99) and Gym Hero (0.97).

This comparison shows the biggest flaw in the scoring recipe. If you ask the system for
high-energy music using a genre and mood that only appear in a slow, quiet song, the
genre and mood bonuses (worth 3.0 points combined) will always win. The energy score
can only add up to 1.0, so it can never overcome a genre+mood match — even when the
energy match is terrible. The system is being logically consistent; the rules just
produce a result that no human listener would find acceptable.

Comparing these two profiles shows when the system works well and when it backfires.
It works well when a listener's genre, mood, and energy preferences all point to the
same type of song. It fails when those preferences conflict with each other or with
what the catalog actually contains.

---

## Pair 3: Ghost Genre (k-pop) vs Ignored Dimensions (classical)

These two adversarial profiles expose opposite problems in the scoring system.

The Ghost Genre profile (genre: k-pop, mood: happy, energy: 0.75) got Rooftop Lights
at #1 with a score of 1.99 — just under 2.0 out of 4.0. Because k-pop is not in the
catalog, no song ever earned the genre bonus. The best this listener could do was find
songs with the right mood and close energy. That 2.0 ceiling means a k-pop fan's best
possible result looks the same as a pop fan's worst result.

The Ignored Dimensions profile (genre: classical, mood: peaceful, energy: 0.22) got
Autumn Sonata No. 3 at a perfect 4.0. On the surface, that looks like the system
working exactly right.

But here is the important contrast: the classical listener also specified acousticness
0.99, valence 0.70, and a specific tempo. The system never checked any of those fields.
Autumn Sonata happened to be the only classical song and the only peaceful song in the
catalog, so it won automatically — the perfect score was inevitable regardless of what
acousticness or tempo the user asked for.

The k-pop listener was penalized for having a genre the catalog does not cover. The
classical listener was rewarded with a perfect score even though three of their
preferences were silently ignored. These are opposite problems — one user gets too
little credit, the other appears to get perfect credit for the wrong reasons.

Both cases point to the same root issue: the scoring system is not actually measuring
how well a song fits a listener. It is measuring how well the catalog's data labels
happen to overlap with the user's inputs. When that overlap is complete, the score
looks perfect. When it is missing entirely, the user is invisible to the system.

---

## Why Does Gym Hero Keep Showing Up for Happy Pop Listeners?

This is worth explaining in plain language, because it came up across multiple tests.

A High-Energy Pop listener says they want "happy" music. Gym Hero is a pop song — so
it earns 2.0 points for genre match right away. That 2.0 is already more than the
maximum you can earn from mood and energy combined (1.0 + 1.0 = 2.0). Even before
the mood is checked, Gym Hero is already tied for the theoretical best a non-pop song
could ever reach.

Then the system checks mood. Gym Hero's mood is "intense," not "happy" — so it earns
nothing there. Then it checks energy. Gym Hero has energy 0.93 and the user wants 0.80,
so it earns 0.87 for energy closeness.

Final score: 2.0 + 0 + 0.87 = 2.87.

Now compare that to Rooftop Lights, which is indie pop (not pop), mood happy (matches!),
and energy 0.76 (close to 0.80). It earns: 0 + 1.0 + 0.96 = 1.96.

Rooftop Lights is a much better match for what the user actually described — it is
happy, it has similar energy, it is clearly in the pop family. But because the genre
label is "indie pop" instead of "pop," it earns zero genre points and loses to Gym Hero
by almost a full point.

Gym Hero keeps showing up not because the system thinks it is happy — it literally does
not give it a single point for mood — but because the system values the word "pop" in
the genre field so heavily that an intense pop song will almost always beat a happy
indie pop song. The genre bonus is like a VIP pass that gets you most of the way to
the top before any other factor is considered.

---

## Overall Reflection

The clearest pattern across from these six profiles is that the system is fairest to users whose taste matches the catalog's most common genres and moods; pop, lofi, and songs with happy, chill, or intense moods. Those users get real variety and meaningful
ranking differences within their preferred group.

Everyone else like listeners with niche genres, conflicting preferences, or tastes that
the catalog's data labels do not capture gets a system that is working correctly by
its own rules but producing results that feel off. The rules are not wrong. The problem
is that the rules were designed around the catalog that exists, not the full range of
human musical taste.

That is the deeper lesson: a recommender system is only as fair as the data it is
built on and the features it chooses to weight. Changing the weights helped at the
margins by narrowing some gaps and corrected some orderings but it could not fix
a catalog that only has one sad song, no moderate-energy songs, and no coverage for
listeners outside a fairly narrow slice of Western popular music.
