"""
Generates a small but fully legal sample PGN dataset by actually playing
out games with python-chess (which enforces legality on every move), so
the dataset ships with zero risk of malformed/illegal SAN -- unlike
hand-typing chess notation, which is easy to get subtly wrong.

Run with: python generate_sample_pgn.py > sample_data/generated_games.pgn
Then merge into sample_data/games.pgn as desired.
"""
import random

import chess
import chess.pgn

OPENING_LINES = {
    "Sicilian Defense": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"],
    "French Defense": ["e4", "e6", "d4", "d5", "Nc3", "Nf6", "e5", "Nfd7"],
    "Caro-Kann Defense": ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5"],
    "Queen's Gambit Declined": ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7"],
    "King's Indian Defense": ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"],
    "English Opening": ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6", "g3", "d5"],
}


def play_random_game(opening_moves, max_plies=44, seed=None):
    rng = random.Random(seed)
    board = chess.Board()
    for san in opening_moves:
        board.push_san(san)

    while not board.is_game_over() and board.ply() < max_plies:
        legal = list(board.legal_moves)
        # Mildly prefer captures/checks to keep games from being totally aimless,
        # while still guaranteeing full legality via python-chess itself.
        weighted = []
        for m in legal:
            weight = 1
            if board.is_capture(m):
                weight += 2
            board.push(m)
            if board.is_check():
                weight += 1
            board.pop()
            weighted.extend([m] * weight)
        move = rng.choice(weighted)
        board.push(move)

    return board


def board_to_game(board, white_name, black_name, eco, opening_name):
    game = chess.pgn.Game()
    game.headers["Event"] = f"{opening_name} Practice"
    game.headers["Site"] = "Generated"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["ECO"] = eco

    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_game_over():
        result = "1/2-1/2"
    else:
        # Ended early on move cap; call it on material balance to keep
        # results varied rather than defaulting every unfinished game to a draw.
        wm = sum(len(board.pieces(pt, chess.WHITE)) for pt in chess.PIECE_TYPES)
        bm = sum(len(board.pieces(pt, chess.BLACK)) for pt in chess.PIECE_TYPES)
        result = "1-0" if wm > bm else "0-1" if bm > wm else "1/2-1/2"

    game.headers["Result"] = result

    node = game
    replay = chess.Board()
    for move in board.move_stack:
        node = node.add_variation(move)
        replay.push(move)

    return game, result


def main():
    eco_map = {
        "Sicilian Defense": "B20",
        "French Defense": "C00",
        "Caro-Kann Defense": "B10",
        "Queen's Gambit Declined": "D30",
        "King's Indian Defense": "E60",
        "English Opening": "A20",
    }
    names = [("Player A", "Player B"), ("Player C", "Player D"), ("Player E", "Player F")]

    out_games = []
    seed = 42
    for opening_name, moves in OPENING_LINES.items():
        white, black = names[seed % len(names)]
        board = play_random_game(moves, max_plies=40, seed=seed)
        game, result = board_to_game(board, white, black, eco_map[opening_name], opening_name)
        out_games.append(game)
        seed += 1

    for game in out_games:
        print(game, end="\n\n")


if __name__ == "__main__":
    main()
