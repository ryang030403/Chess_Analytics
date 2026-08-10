"""
Recursive move-tree analysis: builds a trie of opening moves across a
dataset of games, where every node aggregates how often that exact move
sequence was played and what happened when it was (white win / black win
/ draw). This is the same underlying structure real "opening explorer"
tools use to answer "what's the win rate after 1.e4 e5 2.Nf3?".
"""
from dataclasses import dataclass, field

from .pgn_parser import ParsedGame

# A small curated lookup so common openings get a human-readable name.
# Matched by the longest move-prefix found in the table.
OPENING_NAMES = {
    ("e4", "e5", "Nf3", "Nc6", "Bc4"): "Italian Game",
    ("e4", "e5", "Nf3", "Nc6"): "Open Game / Italian-Ruy setups",
    ("e4", "e5", "f4"): "King's Gambit",
    ("e4", "e5", "Nf3", "d6"): "Philidor Defense",
    ("e4", "e5"): "Open Game",
    ("e4", "c5", "Nf3", "d6"): "Sicilian Najdorf family",
    ("e4", "c5"): "Sicilian Defense",
    ("e4", "e6"): "French Defense",
    ("e4", "c6"): "Caro-Kann Defense",
    ("d4", "d5", "c4", "e6"): "Queen's Gambit Declined",
    ("d4", "d5", "c4"): "Queen's Gambit",
    ("d4", "d5"): "Queen's Pawn Game",
    ("d4", "Nf6", "c4", "g6"): "King's Indian / Grunfeld family",
    ("d4", "Nf6"): "Indian Defense",
    ("c4", "e5"): "English Opening: Reversed Sicilian",
    ("c4",): "English Opening",
}


def name_opening(moves_san: list[str]) -> str:
    """Return the most specific known opening name matching this move prefix."""
    best_match = "Unclassified Opening"
    best_len = 0
    for prefix, name in OPENING_NAMES.items():
        n = len(prefix)
        if tuple(moves_san[:n]) == prefix and n > best_len:
            best_match, best_len = name, n
    return best_match


@dataclass
class MoveNode:
    move: str | None  # None for the root node
    games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    children: dict = field(default_factory=dict)  # move_san -> MoveNode

    @property
    def white_win_rate(self) -> float:
        return round(self.white_wins / self.games, 4) if self.games else 0.0

    @property
    def black_win_rate(self) -> float:
        return round(self.black_wins / self.games, 4) if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return round(self.draws / self.games, 4) if self.games else 0.0


class MoveTree:
    """Recursive trie over opening move sequences, aggregating outcome stats
    at every depth. `max_depth` caps how many plies deep the tree grows,
    since full-game trees over large datasets would explode combinatorially."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.root = MoveNode(move=None)

    def add_game(self, game: ParsedGame) -> None:
        self._insert(self.root, game.moves_san, 0, game.result)

    def _insert(self, node: MoveNode, moves: list[str], depth: int, result: str) -> None:
        self._record_result(node, result)
        if depth >= self.max_depth or depth >= len(moves):
            return
        move = moves[depth]
        if move not in node.children:
            node.children[move] = MoveNode(move=move)
        self._insert(node.children[move], moves, depth + 1, result)

    @staticmethod
    def _record_result(node: MoveNode, result: str) -> None:
        node.games += 1
        if result == "1-0":
            node.white_wins += 1
        elif result == "0-1":
            node.black_wins += 1
        elif result == "1/2-1/2":
            node.draws += 1

    def build(self, games: list[ParsedGame]) -> "MoveTree":
        for game in games:
            self.add_game(game)
        return self

    def path_stats(self, moves_san: list[str]) -> MoveNode | None:
        """Look up the tree node reached by following this exact move sequence."""
        node = self.root
        for move in moves_san:
            if move not in node.children:
                return None
            node = node.children[move]
        return node

    def most_popular_lines(self, top_n: int = 5) -> list[tuple]:
        """Return the top_n most-played leaf-ish branches at max depth,
        each as (move_sequence, MoveNode)."""
        results = []

        def walk(node: MoveNode, path: list[str], depth: int):
            if depth > 0:
                results.append((tuple(path), node))
            if depth >= self.max_depth:
                return
            for child_move, child in node.children.items():
                walk(child, path + [child_move], depth + 1)

        walk(self.root, [], 0)
        results.sort(key=lambda r: r[1].games, reverse=True)
        return results[:top_n]


def opening_statistics(games: list[ParsedGame]) -> dict:
    """Aggregate win/draw/loss statistics grouped by named opening."""
    stats: dict[str, dict] = {}
    for game in games:
        name = name_opening(game.moves_san)
        if name not in stats:
            stats[name] = {"games": 0, "white_wins": 0, "black_wins": 0, "draws": 0}
        s = stats[name]
        s["games"] += 1
        if game.result == "1-0":
            s["white_wins"] += 1
        elif game.result == "0-1":
            s["black_wins"] += 1
        elif game.result == "1/2-1/2":
            s["draws"] += 1

    for s in stats.values():
        g = s["games"]
        s["white_win_rate"] = round(s["white_wins"] / g, 4) if g else 0.0
        s["black_win_rate"] = round(s["black_wins"] / g, 4) if g else 0.0
        s["draw_rate"] = round(s["draws"] / g, 4) if g else 0.0

    return dict(sorted(stats.items(), key=lambda kv: kv[1]["games"], reverse=True))
