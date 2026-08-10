#!/usr/bin/env python3
"""
CLI for the Advanced Chess Analytics Engine.

Usage:
    python cli.py --pgn sample_data/games.pgn
"""
import argparse
import time

from engine import load_games, MoveTree, opening_statistics, WinPredictor, material_balance
import chess


def main():
    parser = argparse.ArgumentParser(description="Advanced Chess Analytics Engine")
    parser.add_argument("--pgn", required=True, help="Path to a PGN file")
    parser.add_argument("--depth", type=int, default=10, help="Max ply depth for the move tree")
    parser.add_argument("--top", type=int, default=5, help="How many top lines/openings to show")
    args = parser.parse_args()

    start = time.time()
    games = load_games(args.pgn)
    elapsed = (time.time() - start) * 1000
    print(f"Parsed {len(games)} games from {args.pgn} in {elapsed:.1f}ms\n")

    # --- Opening statistics ---
    print("=== Opening Statistics ===")
    stats = opening_statistics(games)
    for name, s in list(stats.items())[: args.top]:
        print(
            f"{name:<40} games={s['games']:<4} "
            f"white_win%={s['white_win_rate']*100:5.1f}  "
            f"black_win%={s['black_win_rate']*100:5.1f}  "
            f"draw%={s['draw_rate']*100:5.1f}"
        )

    # --- Recursive move-tree: most popular lines ---
    print(f"\n=== Most Popular Move Sequences (depth<= {args.depth}) ===")
    tree = MoveTree(max_depth=args.depth).build(games)
    for path, node in tree.most_popular_lines(top_n=args.top):
        moves_str = " ".join(path)
        print(f"{moves_str:<40} games={node.games:<4} white_win%={node.white_win_rate*100:5.1f}")

    # --- Win probability prediction ---
    print("\n=== Win Probability Model ===")
    predictor = WinPredictor().fit(games)
    for material_diff in [-4, -1, 0, 1, 4]:
        result = predictor.predict(material_diff=material_diff, ply=20, white_to_move=True)
        print(f"material_diff={material_diff:+d}  ->  P(white wins)={result['white_win_probability']:.3f}  [{result['source']}]")

    # Also demonstrate predicting from a real position reached from the sample PGN.
    if games:
        board = chess.Board()
        sample_moves = games[0].moves_san[:10]
        for san in sample_moves:
            board.push_san(san)
        diff = material_balance(board)
        result = predictor.predict(material_diff=diff, ply=len(sample_moves), white_to_move=board.turn == chess.WHITE)
        print(
            f"\nAfter {' '.join(sample_moves)} (material_diff={diff:+d}): "
            f"P(white wins)={result['white_win_probability']:.3f} [{result['source']}]"
        )


if __name__ == "__main__":
    main()
