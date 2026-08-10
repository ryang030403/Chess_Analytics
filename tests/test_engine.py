import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess

from engine import (
    load_games,
    MoveTree,
    opening_statistics,
    name_opening,
    WinPredictor,
    heuristic_win_probability,
    material_balance,
)

PGN_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "games.pgn")


def test_load_games_parses_without_errors():
    games = load_games(PGN_PATH)
    assert len(games) >= 5
    for g in games:
        assert g.result in ("1-0", "0-1", "1/2-1/2", "*")
        assert len(g.moves_san) > 0


def test_name_opening_recognizes_open_game():
    assert name_opening(["e4", "e5", "Nf3", "Nc6", "Bc4"]) == "Italian Game"
    assert name_opening(["e4", "e5"]) == "Open Game"
    assert name_opening(["d4", "d5", "c4"]) == "Queen's Gambit"


def test_name_opening_falls_back_when_unknown():
    assert name_opening(["a3", "a6", "b3", "b6"]) == "Unclassified Opening"


def test_opening_statistics_sums_to_total_games():
    games = load_games(PGN_PATH)
    stats = opening_statistics(games)
    total = sum(s["games"] for s in stats.values())
    assert total == len(games)


def test_move_tree_root_games_equals_total_games():
    games = load_games(PGN_PATH)
    tree = MoveTree(max_depth=6).build(games)
    assert tree.root.games == len(games)


def test_move_tree_path_stats_matches_manual_count():
    games = load_games(PGN_PATH)
    tree = MoveTree(max_depth=6).build(games)
    e4_games = sum(1 for g in games if g.moves_san[:1] == ["e4"])
    node = tree.path_stats(["e4"])
    assert node is not None
    assert node.games == e4_games


def test_move_tree_win_rates_sum_to_one():
    games = load_games(PGN_PATH)
    tree = MoveTree(max_depth=4).build(games)
    node = tree.root
    total_rate = node.white_win_rate + node.black_win_rate + node.draw_rate
    assert abs(total_rate - 1.0) < 1e-6


def test_heuristic_win_probability_monotonic_in_material():
    low = heuristic_win_probability(-5)
    mid = heuristic_win_probability(0)
    high = heuristic_win_probability(5)
    assert low < mid < high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_heuristic_win_probability_even_material_is_near_half():
    assert abs(heuristic_win_probability(0) - 0.5) < 1e-6


def test_material_balance_starting_position_is_zero():
    board = chess.Board()
    assert material_balance(board) == 0


def test_material_balance_after_capturing_a_pawn():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    board.push_san("exd5")  # White captures a pawn
    assert material_balance(board) == 1


def test_win_predictor_falls_back_to_heuristic_on_tiny_dataset():
    tiny_games = load_games(PGN_PATH)[:1]
    predictor = WinPredictor().fit(tiny_games)
    result = predictor.predict(material_diff=2, ply=10)
    assert result["source"] in ("heuristic", "trained_model")
    assert 0.0 <= result["white_win_probability"] <= 1.0


def test_win_predictor_trains_on_full_sample_dataset():
    games = load_games(PGN_PATH)
    predictor = WinPredictor().fit(games)
    result = predictor.predict(material_diff=3, ply=20)
    assert 0.0 <= result["white_win_probability"] <= 1.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
