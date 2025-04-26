import itertools
import re
from abc import ABC, abstractmethod
from collections import Counter


class GameplayError(Exception):
    """
    Throw this error when the model has made a mistake (e.g. played an invalid move). The model will be prompted to
    make another move. If the game should end, throw a RuntimeError instead.
    """
    pass


class Game(ABC):
    @property
    @abstractmethod
    def valid_players(self) -> tuple[int, ...]:
        """Which player IDs are allowed (e.g. (0,1))."""
        raise NotImplementedError()

    @abstractmethod
    def whose_turn(self) -> int: ...

    @abstractmethod
    def is_over(self) -> bool: ...

    @abstractmethod
    def scores(self) -> dict[int, float]: ...

    @abstractmethod
    def state(self, player_id: int | None = None) -> str: ...

    '''
    Returns a string representation of the game state.
    This is the only context the model receives about the game, so make sure to include all necessary details, 
    including a brief summary of the rules and objective (on first turn), instructions for the current turn, 
    current board state, scores, any face-up cards, etc. For games with constrained action spaces, you may consider 
    printing an exhaustive list of available actions to the current player at the end of the state.
    If player_id is None, returns a global state.
    If player_id is not None, returns a state for the given player.
    '''

    @abstractmethod
    def play(self, move: str, player_id: int = 0) -> None: ...

    '''
    Update the game state according to the move. The move string expected here should be consistent with instructions 
    provided to the model in the state function. If the move is invalid, throw a GameplayError. 
    The error should describe why the move is invalid.
    '''


class Piece:
    def __init__(self, owner: int, rank: str):
        self.owner = owner  # 0 or 1
        self.rank = rank  # '2'–'10', 'S' for spy, 'B' for bomb, 'F' for flag


class Cell:
    def __init__(self, is_lake: bool = False, piece: Piece | None = None):
        self.is_lake = is_lake
        self.piece = piece  # None if empty

    def __str__(self):
        if self.is_lake:
            return 'L'
        elif self.piece is None:
            return '.'
        else:
            return self.piece.rank


class Stratego(Game):
    captured: Piece | None
    last_move: dict[int, tuple[tuple[int, int], tuple[int, int]] | None]
    bounce_count: dict[int, int]
    move_history_queue: list[dict]
    board_state_queues: dict[int | None, list[list[str]]]

    @property
    def valid_players(self) -> tuple[int, ...]:
        return 0, 1

    def __init__(self, setup_p0: list[list[str]], setup_p1: list[list[str]], show_board_labels: bool = True):
        self.show_board_labels = show_board_labels
        self.captured = None
        self.first_states = {0: True, 1: True}
        self.board = [[Cell() for _ in range(10)] for _ in range(10)]
        coords_lakes = [(4, 2), (4, 3), (5, 2), (5, 3),
                        (4, 6), (4, 7), (5, 6), (5, 7)]
        for coord in coords_lakes:
            self.board[coord[0]][coord[1]].is_lake = True

        # track each player’s last move and how many times they’ve bounced
        self.last_move = {0: None, 1: None}  # each is (src, dst) or None
        self.bounce_count = {0: 0, 1: 0}

        self.move_history_queue = []  # each entry: dict with keys player, src, dst, battle_msg
        self.board_state_queues = {0: [], 1: [], None: []}  # one queue for each player

        self._current_player = 0

        self._setup_pieces(0, setup_p0)
        self._setup_pieces(1, setup_p1)

        self.board_state_queues[0].append(self._board_lines(0, self.show_board_labels))
        self.board_state_queues[1].append(self._board_lines(1, self.show_board_labels))
        self.board_state_queues[None].append(self._board_lines(None, self.show_board_labels))

    def _setup_pieces(self, player_id: int, setup: list[list[str]]) -> None:
        """
        Place the 40-piece setup for player_id onto self.board.
        `setup` must be a list of 4 lists, each of length 10, containing ranks like
        '10','9','8',…,'2','S','B','F'.  For player 0 we lay them out on rows 6→9
        in order; for player 1 on rows 3→0 but each row reversed.
        Raises GameplayError on any validation failure.
        """
        # 1) Basic validations
        if player_id not in self.valid_players:
            raise GameplayError(f"Invalid player_id {player_id!r}.")
        if not (isinstance(setup, list) and len(setup) == 4 and all(
                isinstance(r, list) and len(r) == 10 for r in setup)):
            raise GameplayError("Setup must be a list of 4 lists, each of length 10.")

        # 2) Flatten and count
        flat = list(itertools.chain.from_iterable(setup))
        counts = Counter(flat)
        expected = {
            '10': 1, '9': 1, '8': 2, '7': 3, '6': 4,
            '5': 4, '4': 4, '3': 5, '2': 8, 'S': 1,
            'F': 1, 'B': 6
        }
        if counts != expected:
            raise GameplayError(
                "Invalid piece counts: "
                f"found {dict(counts)}, expected {expected}."
            )

        # 3) Place onto board
        for row_idx, row_setup in enumerate(setup):
            if player_id == 0:
                board_row = 6 + row_idx
                row_data = row_setup
            else:
                board_row = 3 - row_idx
                row_data = row_setup[::-1]

            for col_idx, rank in enumerate(row_data):
                cell = self.board[board_row][col_idx]
                if cell.is_lake:
                    raise GameplayError(f"Cannot place piece on lake at ({board_row},{col_idx}).")
                # Assuming cell.piece is None beforehand
                cell.piece = Piece(owner=player_id, rank=rank)

    @staticmethod
    def _cell_to_indices(cell: str, player_id: int) -> tuple[int, int]:
        """
        Convert a cell like 'b2' into (row_index, col_index) on self.board.
        Columns a–j → 0–9, rows 1–10 → indices 9–0 respectively.
        If player_id == 1, rotate indices 180° so that player 1’s perspective maps correctly.
        """
        if not isinstance(cell, str):
            raise GameplayError(f"Cell must be a string like 'b2', got {cell!r}")
        m = re.fullmatch(r'([a-jA-J])(10|[1-9])', cell)
        if not m:
            raise GameplayError(f"Invalid cell notation: {cell!r}. Use files a–j and ranks 1–10.")
        col_letter, row_str = m.groups()
        col = ord(col_letter.lower()) - ord('a')
        row = 10 - int(row_str)
        # Rotate for player 1
        if player_id == 1:
            row = 9 - row
            col = 9 - col
        return row, col

    @staticmethod
    def _indices_to_cell(indices: tuple[int, int], player_id: int | None) -> str:
        """
        Convert a board index tuple (row, col) into a cell string like 'b2'.
        If player_id == 1, first rotate the indices 180° so that row/col match
        Player 1’s view.
        """
        r, c = indices
        # rotate for player 1
        if player_id == 1:
            r = 9 - r
            c = 9 - c
        # file letter a–j
        letter = chr(ord('a') + c)
        # rank number 1–10
        rank = 10 - r
        return f"{letter}{rank}"

    def _oriented_board(self, player_id):
        # Player 0 sits at the "bottom" of self.board, the global view assumes player 0's view
        # Player 1 sits at the “bottom” of their view, which is the top of self.board, so rotate cells and pieces
        # 180 degrees

        if player_id == 0 or (player_id is None and self._current_player == 0):
            return self.board
        elif player_id == 1 or (player_id is None and self._current_player == 1):
            rotated_board = [list(reversed(row)) for row in reversed(self.board)]
            return rotated_board

    def _is_first_state(self, player_id):
        if player_id is None:
            return self.first_states[0] or self.first_states[1]
        elif player_id == 0:
            return self.first_states[0]
        elif player_id == 1:
            return self.first_states[1]

    def _has_valid_moves(self, player_id: int) -> bool:
        """
        Check whether player_id has any legal move available.
        """

        for r in range(10):
            for c in range(10):
                cell = self.board[r][c]
                p = cell.piece
                # must own piece and not be immobile
                if p is None or p.owner != player_id or p.rank in ('B', 'F'):
                    continue
                # check four cardinal directions
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    # in bounds
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        tgt = self.board[nr][nc]
                        # valid if target is not lake and is empty or holds enemy
                        if not tgt.is_lake and (tgt.piece is None or tgt.piece.owner != player_id):
                            # move is not valid if it's the same as previous move, and we have a bounce count of 2
                            if self.bounce_count[player_id] == 2:
                                move = ((r, c), (nr, nc))
                                last = self.last_move[player_id]
                                if last is not None and move == (last[1], last[0]):
                                    # If we're moving a Scout, we need to check the next tile, also, in the case
                                    # that the current tile is invalidated because of bounce_count
                                    if p.rank == '2':
                                        nr, nc = r + 2 * dr, c + 2 * dc
                                        if 0 <= nr < 10 and 0 <= nc < 10:
                                            tgt = self.board[nr][nc]
                                            if not tgt.is_lake and (tgt.piece is None or tgt.piece.owner != player_id):
                                                return True
                                    continue
                            return True
        return False

    def _board_lines(self, player_id: int | None = None, show_board_labels: bool = True):
        lines = []
        # Column labels and vertical gap if requested
        if show_board_labels:
            indent = " " * 6
            cols = indent + " ".join(chr(ord('a') + i).ljust(2) for i in range(10))
            lines.append(cols)
            lines.append("")

        # Oriented board rows
        board = self._oriented_board(player_id)
        for idx, row in enumerate(board):
            prefix = ""
            if show_board_labels:
                row_label = str(10 - idx).ljust(3)
                padding_after_label = " " * 2
                prefix = row_label + padding_after_label

            cell_strs = []
            for cell in row:
                if cell.is_lake:
                    sym = 'L'
                elif cell.piece is None:
                    sym = '.'
                else:
                    if player_id is not None and cell.piece is not None and cell.piece.owner != player_id:
                        sym = '?'
                    elif cell.piece is not None:
                        sym = cell.piece.rank
                cell_strs.append(sym.ljust(2))
            line = prefix + (' ' if prefix else '') + ' '.join(cell_strs)
            lines.append(line)
        return lines

    def _get_move_history_summaries(self, player_id: int | None = 0):
        if len(self.move_history_queue) == 0:
            return None, None
        elif len(self.move_history_queue) == 1:
            if player_id == 0 or player_id is None:
                opponent_move_info = None
                own_previous_info = self.move_history_queue[0]
            else:
                opponent_move_info = self.move_history_queue[0]
                own_previous_info = None
        else:
            own_previous_info = self.move_history_queue[0]
            opponent_move_info = self.move_history_queue[1]

        previous_move_summary = None if own_previous_info is None else self._parse_move_history(own_previous_info,
                                                                                                player_id)
        opponent_move_summary = None if opponent_move_info is None else self._parse_move_history(opponent_move_info,
                                                                                                 player_id)
        return previous_move_summary, opponent_move_summary

    def _parse_move_history(self, move_info, player_id: int | None = 0):
        summary = [""]
        owner = move_info['player_id']
        src_cell = self._indices_to_cell(move_info['src'], player_id)
        dst_cell = self._indices_to_cell(move_info['dst'], player_id)

        # Regular move, no battle
        if move_info['dst_rank'] is None:
            piece = move_info['src_rank'] if owner == player_id else '?'

            if player_id is None:
                previous_turn = (f"In the previous turn, Player {owner} moved his {piece} from {src_cell} to "
                                 f"{dst_cell}. ")
            elif owner == player_id:
                previous_turn = f"In your previous turn, you moved your {piece} from {src_cell} to {dst_cell}. "
            else:
                previous_turn = f"In his previous turn, your opponent moved his {piece} from {src_cell} to {dst_cell}. "
            summary.append(previous_turn)
        else:
            attacking_piece = move_info['src_rank']
            defending_piece = move_info['dst_rank']
            if player_id is None:
                previous_turn = (f"In the previous turn, Player {owner} moved his {attacking_piece} from {src_cell} to "
                                 f"{dst_cell}, attacking Player {1 - owner}'s {defending_piece} in that cell. ")
            elif owner == player_id:
                previous_turn = (f"In your previous turn, you moved your {attacking_piece} from {src_cell} to "
                                 f"{dst_cell}, attacking the opponent's {defending_piece} in that cell. ")
            else:
                previous_turn = (f"In his previous turn, your opponent moved his {attacking_piece} from {src_cell} to "
                                 f"{dst_cell}, attacking your {defending_piece} in that cell. ")
            summary.append(previous_turn)

            if move_info['winner'] is None:
                win = "Since both pieces are of the same rank, both of them were removed from the board. "
            elif player_id is None:
                winning_piece = move_info['winning_rank']
                win = (f"Since Player {move_info['winner']}'s {winning_piece} beats Player {1 - move_info['winner']}'s "
                       f"unit, Player {1 - move_info['winner']}'s unit was removed from the board. ")
            elif move_info['winner'] == player_id:
                winning_piece = move_info['src_rank'] if owner == player_id else move_info['dst_rank']
                losing_piece = move_info['dst_rank'] if owner == player_id else move_info['src_rank']
                win = (f"Since your {winning_piece} beats the opponent's {losing_piece}, the opponent's {losing_piece} "
                       f"was removed from the board. ")
            elif move_info['winner'] == 1 - player_id:
                winning_piece = move_info['dst_rank'] if owner == player_id else move_info['src_rank']
                losing_piece = move_info['src_rank'] if owner == player_id else move_info['dst_rank']
                win = (f"Since the opponent's {winning_piece} beats your {losing_piece}, your {losing_piece} "
                       f"was removed from the board. ")
            summary.append(win)

        summary.append("The resulting board from this turn was: ")
        summary.append("")
        return summary

    def whose_turn(self) -> int:
        return self._current_player

    def is_over(self) -> bool:
        # return True if a flag’s been captured or no movable pieces remain
        if self.captured is not None:
            return True

        player0_valid_moves = self._has_valid_moves(0)
        player1_valid_moves = self._has_valid_moves(1)
        return not (player0_valid_moves and player1_valid_moves)

    def scores(self) -> dict[int, float]:
        if self.captured is not None:
            winner = 1 - self.captured.owner
            return {winner: 1.0, self.captured.owner: 0.0}

        player0_valid_moves = self._has_valid_moves(0)
        player1_valid_moves = self._has_valid_moves(1)

        if not player0_valid_moves:
            return {0: 0.0, 1: 1.0}
        elif not player1_valid_moves:
            return {0: 1.0, 1: 0.0}
        else:
            return {0: 0.0, 1: 0.0}

    def state(self, player_id: int | None = None) -> str:
        """
        Returns a string representation of the game state.
        This is the only context the model receives about the game, so make sure to include all necessary details,
        including a brief summary of the rules and objective (on first turn), instructions for the current turn,
        current board state, scores, any face-up cards, etc. For games with constrained action spaces, you may consider
        printing an exhaustive list of available actions to the current player at the end of the state.
        If player_id is None, returns a global state.
        If player_id is not None, returns a state for the given player.
        """
        lines: list[str] = ["Stratego — the goal is to capture the enemy Flag.", ""]
        # Always print header/context

        if self._is_first_state(player_id):
            lines.append("Quick summary of the rules: ")
            lines.append("Units can move one tile in each of the cardinal directions. Scouts represented as '2' "
                         "can move multiple tiles as long as there are no lakes or units in the intermediate tiles. ")
            lines.append("A piece cannot move back and forth between the same two squares in three consecutive turns.")
            lines.append("Only one piece can be moved on a turn.")
            lines.append("If you move into a cell containing an enemy unit, it means you are attacking this unit.")
            lines.append("Bombs 'B' and flags 'F' can't be moved. Bombs beat all units in a fight, except 3's who can "
                         "dismantle the bombs. ")
            lines.append("Spies 'S' can beat the strongest unit 10 if they attack the 10. If the 10 attacks the spy, "
                         "the spy loses. ")
            lines.append("For all other units the greater rank always wins (e.g. 7 beats 6), if two units with the "
                         "same rank fight, both are removed from the game. ")
            lines.append("The player who captures the opponent's Flag 'F' wins the game. If a player at any moment no"
                         "longer has any valid moves, he loses the game. ")
            lines.append("")

        lines.append(
            "Cells are represented by file/columns (a–j) and rank/rows (1–10) and may contain "
            "'.' empty, 'L' lake, 'B' bomb, 'F' flag, numbers '1-10' for units of that rank, "
            "'S' for spy, and '?' for hidden enemy piece."
        )
        lines.append(
            "Specify your move in the form 'cell_to_move_from-cell_to_move_to'. "
            "For example: 'b2-c3', meaning move the unit in cell b2 to c3. \n"
            "Note how scout moves can span across multiple cells (e.g. 'b2-b5') as long as there are no lakes "
            "or other units in the intermediate cells (in 'b3' and 'b4' in the example). \n"
            "You can only move into cells that are empty '.' or are occupied by an enemy unit '?'. \n"
            "You cannot move your own unit onto a cell that contains another one of your own units. "
            "If you move into a cell containing an enemy unit, it means you are attacking this unit. \n"
            "Lakes 'L' are part of the environment and cannot be moved and cannot be moved into or jumped over. "
        )
        if self._is_first_state(player_id):
            lines.append("")

        current_board = None
        if len(self.board_state_queues[player_id]) == 2:
            current_board = self.board_state_queues[player_id][1]
        previous_board = self.board_state_queues[player_id][0]

        previous_move_summary, opponent_move_summary = self._get_move_history_summaries(player_id)

        if previous_move_summary is not None:
            lines.extend(previous_move_summary)

        lines.extend(previous_board)

        if opponent_move_summary is not None:
            lines.extend(opponent_move_summary)

        if current_board is not None:
            lines.extend(current_board)

        done = self.is_over()
        if done:
            if self.captured is not None:
                # someone captured the flag
                winner = 1 - self.captured.owner
                loser = self.captured.owner
                win_msg = (f"\nPlayer {winner} captured Player {loser}'s Flag and won the game! "
                           "The game is now over.")
            else:
                # someone ran out of moves
                m0 = self._has_valid_moves(0)
                if not m0:
                    winner, loser = 1, 0
                else:
                    winner, loser = 0, 1
                win_msg = (f"\nPlayer {loser} has no valid moves left, so Player {winner} won the game! "
                           "The game is now over.")
            lines.append(win_msg)
        else:
            if player_id is None:
                lines.append(f"\nIt is Player {self.whose_turn()}'s turn.")
            else:
                lines.append(f"\nIt is Player {self.whose_turn()}'s (your) turn. Please specify your move. ")

        result = "\n".join(lines) + "\n"
        return result

    def play(self, move: str, player_id: int = 0) -> None:
        """
        Execute a move given in 'b2-c3' format.
        - Parses source and destination using _cell_to_indices.
        - Checks basic legality (ownership, no lakes, bombs/flags immobile).
        - Enforces adjacency (except Scouts '2' can move any distance in cardinal lines).
        - Path for Scouts must be clear (excluding destination, which may hold enemy).
        - Executes simple move or placeholder battle resolution.
        - Switches turn if move succeeds.
        """
        # Parse move string
        try:
            src_str, dst_str = move.split('-')
        except ValueError:
            raise GameplayError(f"Invalid move format: {move!r}. Use 'b2-c3'.")

        src_r, src_c = self._cell_to_indices(src_str, player_id)
        dst_r, dst_c = self._cell_to_indices(dst_str, player_id)

        src_cell = self.board[src_r][src_c]
        dst_cell = self.board[dst_r][dst_c]

        # Validate source
        if src_cell.piece is None:
            raise GameplayError(f"No piece at source {src_str!r}.")
        piece = src_cell.piece
        if piece.owner != player_id:
            raise GameplayError(f"Cannot move opponent's piece at {src_str!r}.")
        if piece.rank in ('B', 'F'):
            raise GameplayError(f"Cannot move immobile piece {piece.rank} at {src_str!r}.")

        # Validate destination
        if dst_cell.is_lake:
            raise GameplayError(f"Cannot move into lake at {dst_str!r}.")
        if dst_cell.piece and dst_cell.piece.owner == player_id:
            raise GameplayError(f"Cannot move onto your own piece at {dst_str!r}.")

        # Movement distance and path
        dr = dst_r - src_r
        dc = dst_c - src_c
        # non-scout must move exactly one cardinal step
        if piece.rank != '2':
            if abs(dr) + abs(dc) != 1:
                raise GameplayError(f"Invalid move distance for piece {piece.rank} from {src_str!r} to {dst_str!r}.")
        else:
            # scout: must move in straight line
            if not (dr == 0 or dc == 0):
                raise GameplayError(f"Invalid Scout move: must be straight line from {src_str!r} to {dst_str!r}.")
            # step through path excluding src and dst
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
            cr, cc = src_r + step_r, src_c + step_c
            while (cr, cc) != (dst_r, dst_c):
                cell_mid = self.board[cr][cc]
                if cell_mid.piece or cell_mid.is_lake:
                    raise GameplayError(f"Path blocked for Scout move at {chr(cc + ord('a'))}{10 - cr}!')")
                cr += step_r
                cc += step_c

        # enforce “no A→B, B→A, A→B” shuttle in 3 straight turns
        current = ((src_r, src_c), (dst_r, dst_c))
        last = self.last_move[player_id]
        if last is not None and current == (last[1], last[0]):
            # this move reverses the player’s immediately prior move
            self.bounce_count[player_id] += 1
        else:
            # either first move or not a direct reversal
            self.bounce_count[player_id] = 1

        if self.bounce_count[player_id] >= 3:
            raise GameplayError(f"Invalid repetitive back-and-forth between {src_str} and {dst_str}: "
                                "you’ve already done that twice in a row. ")
        self.last_move[player_id] = current

        move_history = {
            'player_id': player_id,
            'src': (src_r, src_c),
            'dst': (dst_r, dst_c),
            'src_rank': src_cell.piece.rank,
            'dst_rank': None if dst_cell.piece is None else dst_cell.piece.rank,
            'winner': None,
            'winning_rank': None,
        }

        # Execute move or battle
        if dst_cell.piece is None:
            # simple move
            dst_cell.piece = piece
            src_cell.piece = None
        else:
            # battle
            if dst_cell.piece.rank == 'F':
                self.captured = dst_cell.piece
                dst_cell.piece = piece
                src_cell.piece = None
            elif dst_cell.piece.rank == 'B':
                if piece.rank == '3':
                    dst_cell.piece = piece
                    src_cell.piece = None
                else:
                    src_cell.piece = None
            elif piece.rank == 'S':
                if dst_cell.piece.rank == '10':
                    dst_cell.piece = piece
                elif dst_cell.piece.rank == 'S':
                    dst_cell.piece = None
                    src_cell.piece = None
                else:
                    src_cell.piece = None
            elif dst_cell.piece.rank == 'S':
                if piece.rank == 'S':
                    dst_cell.piece = None
                    src_cell.piece = None
                else:
                    # numeric (or any) attacker beats a spy
                    dst_cell.piece = piece
                    src_cell.piece = None
            elif int(piece.rank) > int(dst_cell.piece.rank):
                dst_cell.piece = piece
                src_cell.piece = None
            elif int(piece.rank) < int(dst_cell.piece.rank):
                src_cell.piece = None
            else:
                dst_cell.piece = None
                src_cell.piece = None

            if dst_cell.piece is None:
                pass  # Keep move_history['winner'] set to None, since both pieces lost and were removed
            else:
                move_history['winner'] = player_id if dst_cell.piece.owner == player_id else 1 - player_id
                move_history['winning_rank'] = dst_cell.piece.rank

        if len(self.move_history_queue) >= 2:
            self.move_history_queue = self.move_history_queue[-1:]
        self.move_history_queue.append(move_history)

        # Switch turn
        self._current_player = 1 - self._current_player
        self.first_states[player_id] = False

        if len(self.board_state_queues[0]) >= 2:
            self.board_state_queues[0] = self.board_state_queues[0][-1:]
        if len(self.board_state_queues[1]) >= 2:
            self.board_state_queues[1] = self.board_state_queues[1][-1:]
        if len(self.board_state_queues[None]) >= 2:
            self.board_state_queues[None] = self.board_state_queues[None][-1:]

        self.board_state_queues[0].append(self._board_lines(0, self.show_board_labels))
        self.board_state_queues[1].append(self._board_lines(1, self.show_board_labels))
        self.board_state_queues[None].append(self._board_lines(None, self.show_board_labels))
