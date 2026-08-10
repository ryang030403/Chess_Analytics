"""
PGN dataset ingestion built on top of the battle-tested `python-chess`
library for move/legality parsing -- this module focuses on turning a raw
PGN file into structured records (moves, result, metadata) ready for the
analytics layer, and can stream arbitrarily large PGN datasets without
loading the whole file into memory at once.
"""
from dataclasses import dataclass, field

import chess
import chess.pgn


@dataclass
class ParsedGame:
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2", "*"
    eco: str
    moves_san: list = field(default_factory=list)  # e.g. ["e4", "e5", "Nf3", ...]


def iter_games(pgn_path: str):
    """Lazily yields ParsedGame records from a PGN file, one game at a time.
    Using python-chess's streaming reader means multi-gigabyte PGN datasets
    never need to be fully materialized in memory."""
    with open(pgn_path, encoding="utf-8", errors="replace") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            moves_san = []
            for move in game.mainline_moves():
                moves_san.append(board.san(move))
                board.push(move)

            yield ParsedGame(
                white=game.headers.get("White", "?"),
                black=game.headers.get("Black", "?"),
                result=game.headers.get("Result", "*"),
                eco=game.headers.get("ECO", ""),
                moves_san=moves_san,
            )


def load_games(pgn_path: str) -> list[ParsedGame]:
    """Eagerly loads all games -- convenient for small/medium datasets and tests."""
    return list(iter_games(pgn_path))
