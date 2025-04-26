import pytest
from classes import Stratego, GameplayError

# Example starting setups for end-to-end simulation
setup_p0 = [
    ['6', 'S', '10', '9', '8', '8', '7', '7', '7', '2'],
    ['6', '6', '6', '5', '5', '5', '5', '4', '2', '4'],
    ['B', '3', '3', '3', '3', '3', '2', '2', '2', 'B'],
    ['B', 'B', '2', '2', 'B', '2', '4', '4', 'B', 'F'],
]
setup_p1 = [
    ['F', 'S', '10', '9', '8', '8', '7', '7', '7', '6'],
    ['6', '6', '6', '5', '5', '5', '5', '4', '4', '4'],
    ['4', '3', '3', '3', '3', '3', '2', '2', '2', '2'],
    ['2', '2', '2', '2', 'B', 'B', 'B', 'B', 'B', 'B'],
]


def test_end_to_end_flag_capture():
    # Initialize game
    game = Stratego(setup_p0, setup_p1)

    # Initial state and turn
    assert not game.is_over(), "Game should not be over at start"
    assert game.whose_turn() == 0, "Player 0 starts"
    state0 = game.state(0)
    assert "Stratego — the goal is to capture the enemy Flag." in state0

    game.play('e4-e5', 0)
    assert game.whose_turn() == 1, "Turn should switch to Player 1"

    game.play('j4-j5', 1)
    assert game.whose_turn() == 0, "Turn should switch back to Player 0"

    # Player 0 captures the flag by moving scout from a6 to a8
    game.play('j4-j7', 0)
    # After capturing flag, game should end
    assert game.is_over(), "Game should be over after flag capture"
    scores = game.scores()
    # Player 0 wins
    assert scores == {0: 1.0, 1: 0.0}, f"Unexpected scores: {scores}"

    # Final state contains win message
    final_state = game.state(0)
    assert "Player 0 captured Player 1's Flag and won the game" in final_state


def test_bounce_rule_prevents_shuttle():
    game = Stratego(setup_p0, setup_p1)
    # Two legal bounces
    game.play('e4-e5', 0)
    game.play('e5-e4', 0)
    # Third bounce should raise
    with pytest.raises(GameplayError) as exc:
        game.play('e4-e5', 0)
    assert 'repetitive back-and-forth' in str(exc.value)


# --- Battle resolution permutations ---

# All ranks to test
ranks = ['F', 'B', 'S'] + [str(i) for i in range(10, 1, -1)] + ['2']
# Only movable attacker ranks
movable_attackers = [r for r in ranks if r not in ('F', 'B')]


def expected_battle(att, defe):
    # Flag: attacker always wins
    if defe == 'F':
        return 'attacker', att
    # Bomb: only 3 defuses
    if defe == 'B':
        if att == '3':
            return 'attacker', '3'
        return 'defender', 'B'
    # Spy attacks
    if att == 'S':
        if defe == '10':
            return 'attacker', 'S'
        if defe == 'S':
            return None, None
        return 'defender', defe
    # Defender spy defending
    if defe == 'S':
        # attacker numeric > spy always wins
        return 'attacker', att
    # Numeric vs numeric
    a_int, d_int = int(att), int(defe)
    if a_int > d_int:
        return 'attacker', att
    if a_int < d_int:
        return 'defender', defe
    return None, None


@pytest.mark.parametrize("att,defe", [(a, d) for a in movable_attackers for d in ranks])
def test_battle_resolution_combinations(att, defe):
    game = Stratego(setup_p0, setup_p1)
    from classes import Piece
    # Clear board except lakes
    for r in range(10):
        for c in range(10):
            cell = game.board[r][c]
            if not cell.is_lake:
                cell.piece = None
    # Place attacker at e4 and defender at e5
    row_att, col_att = game._cell_to_indices('e4', 0)
    row_def, col_def = game._cell_to_indices('e5', 0)
    game.board[row_att][col_att].piece = Piece(0, att)
    game.board[row_def][col_def].piece = Piece(1, defe)

    # Perform attack
    game.play('e4-e5', 0)

    # Check outcome
    final_cell = game.board[row_def][col_def]
    winner, rank = expected_battle(att, defe)
    if winner is None:
        assert final_cell.piece is None, f"Expected both removed for {att} vs {defe}"
    elif winner == 'attacker':
        assert final_cell.piece is not None and final_cell.piece.owner == 0
        assert final_cell.piece.rank == rank
    else:
        assert final_cell.piece is not None and final_cell.piece.owner == 1
        assert final_cell.piece.rank == rank


def test_movement_rules():
    from classes import Piece
    # Define moves to test: move string and description
    moves = {
        'up1': ('e3-e4',),
        'down1': ('e3-e2',),
        'right1': ('e3-f3',),
        'left1': ('e3-d3',),
        'up2': ('e3-e5',),
        'up_far': ('e3-e10',),
        'down_far': ('e3-e1',),
        'diag': ('e3-f4',)
    }
    for rank in ranks:
        for desc, (move,) in moves.items():
            game = Stratego(setup_p0, setup_p1)
            # clear board except lakes
            for r in range(10):
                for c in range(10):
                    if not game.board[r][c].is_lake:
                        game.board[r][c].piece = None
            # place piece at e3
            r3, c3 = game._cell_to_indices('e3', 0)
            game.board[r3][c3].piece = Piece(0, rank)
            # For scout path-block case, place a blocker at e5
            if rank == '2' and desc == 'up_far':
                r5, c5 = game._cell_to_indices('e5', 0)
                game.board[r5][c5].piece = Piece(1, '3')
                expect_ok = False
            else:
                if rank in ('F', 'B'):
                    expect_ok = False
                elif rank == '2':
                    # scout: up2 (e3-e5) should pass, up_far without blocker pass
                    if desc in ('up1', 'down1', 'right1', 'left1', 'up2', 'up_far', 'down_far'):
                        expect_ok = True
                    else:
                        expect_ok = False
                else:
                    # non-scout non-flag/bomb: only adjacent
                    if desc in ('up1', 'down1', 'right1', 'left1'):
                        expect_ok = True
                    else:
                        expect_ok = False
            if expect_ok:
                # should not raise
                game.play(move, 0)
            else:
                with pytest.raises(GameplayError):
                    game.play(move, 0)


# --- Invalid-move error tests ---

def test_invalid_move_empty_source():
    game = Stratego(setup_p0, setup_p1)
    # pick an empty cell (e.g. a5) from initial setup
    with pytest.raises(GameplayError) as exc:
        game.play('a5-a6', 0)
    assert 'No piece at source' in str(exc.value)


def test_invalid_move_into_lake():
    from classes import Piece
    game = Stratego(setup_p0, setup_p1)
    # clear board except lakes and place a scout at c5
    for r in range(10):
        for c in range(10):
            if not game.board[r][c].is_lake:
                game.board[r][c].piece = None
    r5, c5 = game._cell_to_indices('c5', 0)
    game.board[r5][c5].piece = Piece(0, '2')
    # lake at c6
    with pytest.raises(GameplayError) as exc:
        game.play('c5-c6', 0)
    assert 'Cannot move into lake' in str(exc.value)


def test_invalid_cell_notation():
    game = Stratego(setup_p0, setup_p1)
    invalid_moves = ['k4-k5', 'a0-a1', 'a11-a12', 'd5-e15', '4a-5a', 'zz', '']
    for mv in invalid_moves:
        with pytest.raises(GameplayError):
            game.play(mv, 0)


def test_move_opponent_piece_raises():
    game = Stratego(setup_p0, setup_p1)
    # a10 holds Player 1's flag at start
    with pytest.raises(GameplayError) as exc:
        game.play('a10-a9', 0)
    assert "Cannot move opponent's piece" in str(exc.value)


def test_game_end_by_no_moves_for_opponent():
    from classes import Piece
    # Initialize and then clear board except lakes
    game = Stratego(setup_p0, setup_p1)
    for r in range(10):
        for c in range(10):
            if not game.board[r][c].is_lake:
                game.board[r][c].piece = None
    # Place a moving unit for P0 at e4 and an immobile bomb for P1 at e6
    r4, c4 = game._cell_to_indices('e4', 0)
    game.board[r4][c4].piece = Piece(0, '5')
    r6, c6 = game._cell_to_indices('e6', 0)
    game.board[r6][c6].piece = Piece(1, 'B')
    # P0 makes a valid adjacent move
    game.play('e4-e5', 0)
    # Now P1 has only a bomb and no valid moves => game over
    assert game.is_over(), "Game should end when opponent has no valid moves"
    scores = game.scores()
    assert scores == {0: 1.0, 1: 0.0}, f"Expected P0 win, got {scores}"


if __name__ == '__main__':
    pytest.main()
