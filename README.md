# Advanced Chess Analytics Engine

A chess analytics engine that parses large PGN datasets, builds a recursive move-tree for opening exploration, computes opening win/draw/loss statistics, and predicts win probability from a position using both a closed-form heuristic and a trained logistic regression model.

## Features

- **PGN dataset parsing** — streams arbitrarily large PGN files game-by-game using `python-chess`, so a multi-gigabyte dataset never needs to be fully loaded into memory (see `engine/pgn_parser.py::iter_games`)
- **Recursive move-tree analysis** — builds a trie over opening move sequences, aggregating win/draw/loss counts at every node (`engine/move_tree.py::MoveTree`)
- **Opening statistics** — classifies games by named opening (Sicilian, French, Queen's Gambit, etc.) and reports win rates per opening
- **Win probability prediction** — a closed-form material-to-win% heuristic (always available) plus an optional trained `LogisticRegression` model fit on positions replayed from the dataset (material balance, ply, side to move)

## Tech Stack

Python · `python-chess` (legal move parsing/validation) · scikit-learn (`LogisticRegression`) · NumPy · pytest

## Getting Started

```bash
pip install -r requirements.txt

python cli.py --pgn sample_data/games.pgn --depth 8 --top 5
```

Sample output:

```
Parsed 9 games from sample_data/games.pgn in 17.5ms

=== Opening Statistics ===
Philidor Defense       games=1  white_win%=100.0  black_win%=0.0  draw%=0.0
...

=== Most Popular Move Sequences (depth<=8) ===
e4          games=5  white_win%=60.0
e4 e5       games=2  white_win%=100.0
...

=== Win Probability Model ===
material_diff=-4  ->  P(white wins)=0.760  [trained_model]
material_diff=+0  ->  P(white wins)=0.730  [trained_model]
material_diff=+4  ->  P(white wins)=0.697  [trained_model]
```

The sample dataset (`sample_data/games.pgn`) includes three real historical games (Morphy's "Opera Game", Anderssen's "Immortal Game", Fischer–Spassky 1972) plus six additional fully-legal games generated programmatically with `generate_sample_pgn.py` — every move is validated by `python-chess` itself, so there's zero risk of malformed notation.

## Running Tests

```bash
python -m pytest tests/ -v
```

13 tests cover PGN parsing correctness, opening classification, move-tree aggregation invariants (win/draw/loss rates summing to 1, node counts matching manual counts), material balance calculation, and win-probability prediction (both the heuristic and trained-model code paths).

## Evaluation on Real Games

`evaluate.py` measures the win-probability model on **793 real Lichess blitz games** (`sample_data/lichess_blitz.pgn`, two public arenas, CC0). The split is by game rather than by position so no game leaks across train and test, and results are averaged over 5 random splits.

```bash
python evaluate.py
```

| Predictor | Held-out accuracy (mean of 5 splits) |
|---|---|
| Trained logistic regression | 61.6% (range 56.9–63.6%) |
| Material-count heuristic (no training) | 61.9% (range 57.0–65.6%) |
| Majority-class baseline | 51.0% |

By game phase, model accuracy is ~50% in the opening (ply 1–20), ~65% in the middlegame and ~77% past ply 60.

**Honest read:** both predictors beat the baseline by about 10 points, but the trained model does not beat the hand-written material heuristic. With only three features (material, ply, side to move), the model mostly re-learns "more material wins". Adding features such as mobility, king safety or an engine evaluation is the obvious next step.

## Usage as a library

```python
from engine import load_games, MoveTree, opening_statistics, WinPredictor

games = load_games("my_games.pgn")

# Opening win-rate breakdown
stats = opening_statistics(games)

# Recursive move-tree — "what's the win rate after 1.e4 e5 2.Nf3?"
tree = MoveTree(max_depth=10).build(games)
node = tree.path_stats(["e4", "e5", "Nf3"])
print(node.games, node.white_win_rate)

# Win probability prediction
predictor = WinPredictor().fit(games)
predictor.predict(material_diff=2, ply=15, white_to_move=True)
```

## Design Notes

- **Why TF-IDF-style heuristic + trainable model, not a neural net?** The heuristic (logistic curve over material balance) is the same family of formula real engines use to convert a centipawn evaluation into a win percentage — it's instant, explainable, and needs zero data. The trained `LogisticRegression` layer improves on top of it as more games are ingested, without ever leaving the system unusable on a small dataset.
- **Why a recursive trie for the move tree instead of a flat table?** Aggregating stats depth-first mirrors how chess opening explorers actually work (Lichess/Chess.com's "explorer" tools use the same underlying structure), and it naturally supports arbitrary-depth prefix lookups (`path_stats`) without recomputation.
- Extending `OPENING_NAMES` in `engine/move_tree.py` with more ECO-mapped prefixes will improve opening classification coverage on larger real-world datasets (e.g. a full Lichess database export).

## License

This project is licensed under the [MIT License](LICENSE).

It depends on [`python-chess`](https://github.com/niklasf/python-chess) (imported as `chess`), which is licensed GPL-3.0+. That dependency is used as-is via pip and none of its code is vendored or modified here.
