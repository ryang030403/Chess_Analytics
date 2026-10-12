"""
Held-out evaluation of the win-probability model on real games.

Dataset: sample_data/lichess_blitz.pgn -- every game from two public Lichess
blitz arenas (tournament IDs Oldc3ktZ and w5ExlW8J), exported via the Lichess
API. Lichess game data is released under CC0. To re-download:

    curl "https://lichess.org/api/tournament/Oldc3ktZ/games?clocks=false&evals=false&opening=false"  > sample_data/lichess_blitz.pgn
    curl "https://lichess.org/api/tournament/w5ExlW8J/games?clocks=false&evals=false&opening=false" >> sample_data/lichess_blitz.pgn

Method:
  * Only decisive games are used (the model predicts P(white wins)).
  * The split is by GAME, not by position, so no game contributes positions
    to both train and test (positions from one game share a label, so a
    position-level split would leak).
  * Every position in the held-out games is scored. Prediction = White wins
    if P >= 0.5.
  * Compared against two baselines: always predicting the training set's
    majority class, and the closed-form material heuristic.

Run with: python evaluate.py [--pgn PATH] [--test-size 0.2] [--seeds 5]
"""
import argparse
import random
import time

import numpy as np

from engine import WinPredictor, load_games
from engine.win_predictor import extract_training_samples, heuristic_win_probability


def evaluate(games, test_size, seed):
    decisive = [g for g in games if g.result in ("1-0", "0-1")]
    rng = random.Random(seed)
    rng.shuffle(decisive)
    cut = int(len(decisive) * (1 - test_size))
    train_games, test_games = decisive[:cut], decisive[cut:]

    model = WinPredictor().fit(train_games)
    assert model.model is not None, "model fell back to heuristic; dataset too small"

    train_y = np.array([s.white_won for s in extract_training_samples(train_games)])
    majority = int(train_y.mean() >= 0.5)

    test = extract_training_samples(test_games)
    X = np.array([[s.material_diff, s.ply, int(s.white_to_move)] for s in test])
    y = np.array([s.white_won for s in test])

    model_pred = (model.model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    heur_pred = np.array([heuristic_win_probability(s.material_diff) >= 0.5 for s in test]).astype(int)

    return {
        "train_games": len(train_games),
        "test_games": len(test_games),
        "test_positions": len(test),
        "model": (model_pred == y).mean(),
        "majority": (y == majority).mean(),
        "heuristic": (heur_pred == y).mean(),
        "by_phase": {
            name: ((model_pred == y)[mask].mean(), int(mask.sum()))
            for name, mask in (
                ("opening (ply 1-20)", X[:, 1] <= 20),
                ("middlegame (21-60)", (X[:, 1] > 20) & (X[:, 1] <= 60)),
                ("endgame (61+)", X[:, 1] > 60),
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pgn", default="sample_data/lichess_blitz.pgn")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, default=5, help="number of random game-level splits to average over")
    args = parser.parse_args()

    t0 = time.perf_counter()
    games = load_games(args.pgn)
    parse_ms = (time.perf_counter() - t0) * 1000
    print(f"Parsed {len(games)} games in {parse_ms:.0f}ms")

    runs = [evaluate(games, args.test_size, seed) for seed in range(args.seeds)]
    r0 = runs[0]
    print(f"Split: {r0['train_games']} train games / {r0['test_games']} held-out games "
          f"(~{r0['test_positions']} held-out positions per split), {args.seeds} random splits\n")

    def summary(key):
        vals = np.array([r[key] for r in runs])
        return f"{vals.mean() * 100:5.1f}%  (min {vals.min() * 100:.1f}%, max {vals.max() * 100:.1f}%)"

    print("Held-out accuracy (mean over splits):")
    print(f"  Trained logistic regression : {summary('model')}")
    print(f"  Material heuristic baseline : {summary('heuristic')}")
    print(f"  Majority-class baseline     : {summary('majority')}")

    print("\nModel accuracy by game phase (split 0):")
    for name, (acc, n) in r0["by_phase"].items():
        print(f"  {name:20s} {acc * 100:5.1f}%  ({n} positions)")


if __name__ == "__main__":
    main()
