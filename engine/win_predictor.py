"""
Win probability prediction.

Two layers, so the engine is always useful even on a tiny dataset:

  1. A closed-form heuristic baseline that converts material balance into
     a win probability via a logistic curve -- the same family of formula
     chess engines commonly use to turn a centipawn evaluation into a
     win percentage. Works instantly with zero training data.

  2. An optional trained `LogisticRegression` (scikit-learn) fit on
     positions replayed from the parsed PGN dataset (material balance,
     ply number, side to move), predicting P(white wins). This is what
     "improves" as more games are fed into the engine.
"""
import math
from dataclasses import dataclass

import chess
import numpy as np
from sklearn.linear_model import LogisticRegression

from .pgn_parser import ParsedGame

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_balance(board: chess.Board) -> int:
    """White material minus Black material, in pawns."""
    balance = 0
    for piece_type, value in PIECE_VALUES.items():
        balance += value * len(board.pieces(piece_type, chess.WHITE))
        balance -= value * len(board.pieces(piece_type, chess.BLACK))
    return balance


def heuristic_win_probability(material_diff: int) -> float:
    """Converts a material advantage (in pawns, white minus black) into a
    win probability for White using a logistic curve. A +4 pawn edge
    (e.g. up a minor piece) maps to roughly an 85% win probability."""
    return round(1 / (1 + math.exp(-0.5 * material_diff)), 4)


@dataclass
class PositionSample:
    material_diff: int
    ply: int
    white_to_move: bool
    white_won: int  # 1 if the game White eventually won, else 0


def extract_training_samples(games: list[ParsedGame]) -> list[PositionSample]:
    """Replays every game move-by-move (using python-chess for legal move
    application) and records a labeled feature sample at each position.
    Draws are excluded from the label but still contribute earlier-ply
    material signal implicitly through other games."""
    samples: list[PositionSample] = []
    for game in games:
        if game.result not in ("1-0", "0-1"):
            continue
        white_won = 1 if game.result == "1-0" else 0

        board = chess.Board()
        for ply, san in enumerate(game.moves_san):
            try:
                move = board.parse_san(san)
            except ValueError:
                break
            board.push(move)
            samples.append(
                PositionSample(
                    material_diff=material_balance(board),
                    ply=ply + 1,
                    white_to_move=board.turn == chess.WHITE,
                    white_won=white_won,
                )
            )
    return samples


class WinPredictor:
    """Predicts P(white wins) for a given position. Falls back to the
    closed-form heuristic when there isn't enough labeled training data
    (or when both classes aren't represented) to fit a stable model."""

    MIN_SAMPLES = 20

    def __init__(self):
        self.model: LogisticRegression | None = None

    def fit(self, games: list[ParsedGame]) -> "WinPredictor":
        samples = extract_training_samples(games)
        if len(samples) < self.MIN_SAMPLES:
            self.model = None
            return self

        X = np.array([[s.material_diff, s.ply, int(s.white_to_move)] for s in samples])
        y = np.array([s.white_won for s in samples])

        if len(set(y.tolist())) < 2:
            # Can't fit a meaningful classifier on a single outcome class.
            self.model = None
            return self

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        return self

    def predict(self, material_diff: int, ply: int, white_to_move: bool = True) -> dict:
        heuristic = heuristic_win_probability(material_diff)
        if self.model is None:
            return {"white_win_probability": heuristic, "source": "heuristic", "trained": False}

        X = np.array([[material_diff, ply, int(white_to_move)]])
        model_prob = round(float(self.model.predict_proba(X)[0][1]), 4)
        return {"white_win_probability": model_prob, "source": "trained_model", "trained": True}
