import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import TypedDict, Optional, Tuple, cast


REQUIRED_PIECES = {'10': 1, '9': 1, '8': 2, '7': 3, '6': 4, '5': 4, '4': 4, '3': 5, '2': 8, 'S': 1, 'F': 1, 'B': 6}
EXPECTED_TOTAL_PIECES = sum(REQUIRED_PIECES.values())  # Should be 40
BOARD_SETUP_ROWS = 4
BOARD_SETUP_COLS = 10


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


class MoveInfo(TypedDict):
    """
    Used for summarizing detailed information about the past two moves
    """
    player_id: int
    src: Tuple[int, int]
    dst: Tuple[int, int]
    src_rank: str
    dst_rank: Optional[str]
    winner: Optional[int]
    winning_rank: Optional[str]
    losing_rank: Optional[str]


class RemovedHistory(TypedDict):
    """
    Used for tracking and displaying all past battle resolutions
    """
    turn: int
    lost_rank: Optional[str]
    by_rank: Optional[str]


class MoveHistory(TypedDict):
    """
    Used for tracking and displaying all past player moves
    """
    turn: int
    move: str


class Piece:
    """
    Represents a single Stratego piece.

    Attributes:
        owner (int): ID of the player who owns this piece (0 or 1).
        rank (str): The rank symbol of the piece:
            - '2'–'10' for Scouts through highest-rank units,
            - 'S' for Spy,
            - 'B' for Bomb,
            - 'F' for Flag.
        numeric_rank (int): The numeric rank of the piece, may not always be suitable for direct comparison, e.g.
            spies beat 10's if attacking first, bombs are beaten by 3's.
    """

    def __init__(self, owner: int, rank: str):
        """
        Initialize a Stratego piece.

        Args:
            owner (int): Player ID (0 or 1).
            rank (str): Rank symbol as described above.
        """
        self.owner = owner  # which player this piece belongs to
        self.rank = rank  # the piece’s combat rank or special symbol

        if self.rank == 'S':
            self.numeric_rank = 1  # Spies are beaten by everything except if attacking a 10 (we handle this explicitly)
        elif self.rank == 'F':
            self.numeric_rank = 0  # Flag is beaten by everything
        elif self.rank == 'B':
            self.numeric_rank = 11  # Bomb beats everything (except 3's, we handle this explicitly)
        else:
            self.numeric_rank = int(self.rank)


class Cell:
    """
    Represents a single square on the Stratego board.

    Attributes:
        is_lake (bool): True if this cell is an impassable lake.
        piece (Piece | None): The piece occupying this cell, or None if empty.
    """

    def __init__(self, is_lake: bool = False, piece: Piece | None = None):
        """
        Initialize a board cell.

        Args:
            is_lake (bool): whether the cell is one of the 8 lake squares.
            piece (Piece | None): optional piece to place here at startup.
        """
        self.is_lake = is_lake
        self.piece = piece  # None if there’s no piece here

    def __str__(self):
        """
        Return a single-character representation for display:
         - 'L' for lake
         - '.' for empty
         - otherwise the piece’s rank (e.g. '5', 'S', 'B', 'F')
        """
        if self.is_lake:
            return 'L'
        elif self.piece is None:
            return '.'
        else:
            return self.piece.rank


class Stratego(Game):
    """
    Implementation of the classic board game Stratego.
    Players set up their 40 pieces on a 10x10 board and take turns moving units
    to attack and capture the opponent's Flag or immobilize them.
    Handles piece placement, move validation (including Scout movement and bounce rules),
    battle resolution, and game-over conditions.
    """

    captured: Piece | None
    last_move: dict[int, tuple[tuple[int, int], tuple[int, int]] | None]
    bounce_count: dict[int, int]
    move_info_queue: list[MoveInfo]
    board_state_queues: dict[int | None, list[list[str]]]
    removed_pieces: dict[int, list[RemovedHistory]]
    move_histories: dict[int, list[MoveHistory]]

    @property
    def valid_players(self) -> tuple[int, ...]:
        """
        Which player IDs are used.
        """
        return 0, 1

    def __init__(self, setup_p0: Optional[list[list[str]]] = None, setup_p1: Optional[list[list[str]]] = None,
                 show_board_labels: bool = True, aggressor_advantage: bool = False, flip_board: bool = False,
                 show_player_labels: bool = True, display_removed_pieces: bool = True, display_past_moves: bool = True):
        """
        Initialize a Stratego game instance.

        Args:
            setup_p0: 4x10 grid of ranks for Player 0 (rows 6→9 on the board).
            setup_p1: 4x10 grid of ranks for Player 1 (rows 3→0, each row reversed).
            show_board_labels: Whether to print file/rank labels alongside the board.
            aggressor_advantage: Extra tournament rule: If True, attacking pieces win if two battling pieces have
            the same rank (e.g. '6' and '6'). If False, both pieces lose and are removed by default.
            flip_board: Whether to flip the board state representation for player 1 so both players view
            the board with their own pieces at the bottom. If False, we always return the internal board perspective.
            show_player_labels: Whether to show 'Player 0' and 'Player 1' labels at the corresponding sides of the
            board state string.
            display_removed_pieces: Whether to display the captured/removed pieces of each player in the state.
            display_past_moves: Whether to display past moves of each player in the state.
        """
        self.show_board_labels = show_board_labels
        self.aggressor_advantage = aggressor_advantage
        self.flip_board = flip_board
        self.show_player_labels = show_player_labels
        self.display_removed_pieces = display_removed_pieces
        self.display_past_moves = display_past_moves
        self.captured = None  # Tracks a captured Flag piece, if any.
        self.first_states = {0: True, 1: True}  # To print full rules on the first view for each player.
        self.removed_pieces = {0: [], 1: []}  # Keep track which pieces were removed for each player
        self.turn_counter = 1

        # Create empty 10x10 board and mark the 8 lake cells
        self.board = [[Cell() for _ in range(10)] for _ in range(10)]
        lake_coords = [(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)]
        for r, c in lake_coords:
            self.board[r][c].is_lake = True

        # Track repetition: last move and bounce count for each player
        self.last_move = {0: None, 1: None}
        self.bounce_count = {0: 0, 1: 0}

        # Queues to build history and board-state diffs for generating state strings
        self.move_info_queue: list[dict] = []
        self.board_state_queues = {0: [], 1: [], None: []}
        self.move_histories = {0: [], 1: []}

        self._current_player = 0

        if setup_p0 is not None:
            # Place pieces for both players; raises GameplayError on invalid setup
            self.setup_pieces(0, setup_p0)

        if setup_p1 is not None:
            self.setup_pieces(1, setup_p1)

        # Only fill queues if board is fully setup already, otherwise it will fill queues in user call to setup_pieces
        if setup_p0 is not None and setup_p1 is not None:
            # Seed the initial board snapshots for each perspective
            for pid in (0, 1, None):
                self.board_state_queues[pid].append(self._board_lines(pid, self.show_board_labels))

    @staticmethod
    def _validate_setup(player_id: int, setup: list[list[str]]) -> Optional[str]:
        """
        Validates a generated setup based on dimensions and piece counts.
        Returns an error message string if invalid, None otherwise.
        """
        if not isinstance(setup, list):
            return f"Invalid setup format for Player {player_id}: Expected a list, got {type(setup).__name__}."

        if len(setup) != BOARD_SETUP_ROWS:
            return f"Invalid setup for Player {player_id}: Expected {BOARD_SETUP_ROWS} rows, got {len(setup)}."

        if not all(isinstance(row, list) for row in setup):
            return f"Invalid setup for Player {player_id}: All elements in the main list must be lists (rows)."

        if not all(len(row) == BOARD_SETUP_COLS for row in setup):
            return f"Invalid setup for Player {player_id}: Each row must have {BOARD_SETUP_COLS} columns."

        # Flatten the list and count pieces
        all_pieces = [piece for row in setup for piece in row]

        if len(all_pieces) != EXPECTED_TOTAL_PIECES:
            return (f"Invalid setup for Player {player_id}: Expected {EXPECTED_TOTAL_PIECES} total pieces, got "
                    f"{len(all_pieces)}.")

        counts = Counter(all_pieces)
        required_keys = set(REQUIRED_PIECES.keys())
        actual_keys = set(counts.keys())

        # Check for unexpected piece symbols
        if not actual_keys.issubset(required_keys):
            unexpected = actual_keys - required_keys
            return f"Invalid setup for Player {player_id}: Found unexpected piece symbols: {sorted(list(unexpected))}."

        # Check for missing piece symbols (that should be present)
        if not required_keys.issubset(actual_keys):
            missing = required_keys - actual_keys
            return f"Invalid setup for Player {player_id}: Missing required piece symbols: {sorted(list(missing))}."

        # Check counts for each piece type
        for piece, required_count in REQUIRED_PIECES.items():
            actual_count = counts.get(piece, 0)
            if actual_count != required_count:
                return (f"Invalid setup for Player {player_id}: Incorrect count for piece '{piece}'. Expected "
                        f"{required_count}, got {actual_count}.")

        return None  # Validation passed

    def setup_pieces(self, player_id: int, setup: list[list[str]]) -> None:
        """
        setup: 4x10 grid of ranks for Player 0 (rows 6→9 on the board), Player 1 (rows 3→0, each row reversed).

        Place the 40-piece setup for player_id onto self.board.
        `setup` must be a list of 4 lists, each of length 10, containing ranks like
        '10','9','8',…,'2','S','B','F'.  For player 0 we lay them out on rows 6→9
        in order; for player 1 on rows 3→0 but each row reversed.
        Raises GameplayError on any validation failure.
        """
        # 1) Basic validations
        if player_id not in self.valid_players:
            raise GameplayError(f"Invalid player_id {player_id!r}.")

        # 2) validate the input board
        val_message = self._validate_setup(player_id, setup)
        if val_message is not None:
            raise GameplayError(val_message)

        # 3) Place pieces onto board
        for row_idx, row_setup in enumerate(setup):
            if player_id == 0:
                board_row = 6 + row_idx
                row_data = row_setup
            else:
                # Reverse for player 1
                board_row = 3 - row_idx
                row_data = row_setup[::-1]

            # Place the pieces by setting cell.piece with the correct rank and owner id
            for col_idx, rank in enumerate(row_data):
                cell = self.board[board_row][col_idx]
                # Assuming cell.piece is None beforehand
                cell.piece = Piece(owner=player_id, rank=rank)

        # Seed the initial board snapshots for each perspective
        for pid in (0, 1, None):
            self.board_state_queues[pid] = []  # first empty queues on each call
            self.board_state_queues[pid].append(self._board_lines(pid, self.show_board_labels))

    def _cell_to_indices(self, cell: str, player_id: int) -> tuple[int, int]:
        """
        Convert cell notation like 'b2' into board indices (row, col).
        Adjusts for Player 1 perspective by rotating 180°.

        Raises GameplayError for invalid notation.
        """
        # Validate input str is in the valid notation
        if not isinstance(cell, str):
            raise GameplayError(f"Cell must be a string like 'b2', got {cell!r}")
        m = re.fullmatch(r'([a-jA-J])(10|[1-9])', cell)
        if not m:
            raise GameplayError(f"Invalid cell notation: {cell!r}. Use files a–j and ranks 1–10.")

        # Extract col and row str and convert to indices
        col_letter, row_str = m.groups()
        col = ord(col_letter.lower()) - ord('a')
        row = 10 - int(row_str)

        # Rotate for player 1
        if self.flip_board and player_id == 1:
            row = 9 - row
            col = 9 - col
        return row, col

    def _indices_to_cell(self, indices: tuple[int, int], player_id: int | None) -> str:
        """
        Convert board indices (row, col) to cell notation like 'b2'.
        Rotates indices for Player 1's perspective if needed.
        """
        r, c = indices
        # rotate for player 1
        if self.flip_board and player_id == 1:
            r = 9 - r
            c = 9 - c
        # file letter a–j
        letter = chr(ord('a') + c)
        # rank number 1–10
        rank = 10 - r
        return f"{letter}{rank}"

    def _oriented_board(self, player_id: int | None):
        """
        Returns the board matrix oriented to the given player's view if self.flip_board is True.
        Player 1's view is a 180° rotation of the internal board.
        If player_id is None, orients the board according to the current player.
        If self.flip_board is False, we return the internal board perspective.
        """
        if not self.flip_board:
            return self.board
        elif player_id == 0 or (player_id is None and self._current_player == 0):
            return self.board
        elif player_id == 1 or (player_id is None and self._current_player == 1):
            rotated_board = [list(reversed(row)) for row in reversed(self.board)]
            return rotated_board

    def _is_first_state(self, player_id: int | None) -> bool:
        """
        Helper to check if this is the first time the player is seeing the state,
        to decide whether to include full rules/context text.
        If player_id is None, it's a first state until both players have made a move.
        """
        if player_id is None:
            return self.first_states[0] or self.first_states[1]
        return self.first_states.get(player_id, False)

    def _has_valid_moves(self, player_id: int) -> bool:
        """
        Checks if any movable piece for player_id has at least one legal move available.
        Considers adjacency, lakes, occupancy, scouts' long moves, and bounce restrictions.
        """
        for r in range(10):
            for c in range(10):
                cell = self.board[r][c]
                p = cell.piece
                # must own piece and not be immobile
                if p is None or p.owner != player_id or p.rank in ('B', 'F'):
                    continue
                # check the four cardinal directions
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    # compute the new cell indices
                    nr, nc = r + dr, c + dc
                    # ensure they're in bounds
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        tgt = self.board[nr][nc]
                        # valid if target is not lake and is empty or holds enemy
                        if not tgt.is_lake and (tgt.piece is None or tgt.piece.owner != player_id):
                            # move is not valid if it's the same as previous move, and we have a bounce count of 2
                            if self.bounce_count[player_id] == 2:
                                move = ((r, c), (nr, nc))
                                last = self.last_move[player_id]
                                if last is not None and move == (last[1], last[0]):
                                    # If we're moving a Scout, we need to check the next tile also (in the case
                                    # that the current tile is invalidated because of bounce_count)
                                    if p.rank == '2':
                                        # set the new indices to the cell two steps in the current cardinal direction
                                        nr, nc = r + 2 * dr, c + 2 * dc
                                        # ensure they're in bounds
                                        if 0 <= nr < 10 and 0 <= nc < 10:
                                            tgt = self.board[nr][nc]
                                            # if it's not a lake, and there's no friendly piece, we can move there
                                            if not tgt.is_lake and (tgt.piece is None or tgt.piece.owner != player_id):
                                                return True
                                    continue
                            return True
        return False

    def _board_lines(self, player_id: int | None = None, show_board_labels: bool = True) -> list[str]:
        """
        Renders the board as a list of text lines for display to the player.
        Hides enemy pieces as '?' and shows lakes, empty, and own ranks appropriately.
        """
        lines = []

        # Define player labels
        if not self.flip_board:
            top_player, bottom_player = 1, 0
        else:
            # If flip_board is True, current player is always on the bottom
            if player_id is None or player_id == 0:
                top_player, bottom_player = 1, 0
            else:
                top_player, bottom_player = 0, 1

        # Add top player label
        padding = " " * 15 if self.show_board_labels else " " * 10
        if self.show_player_labels:
            lines.append(padding + f"Player {top_player}")

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
                    sym = '.'  # Empty are '.'
                else:
                    if player_id is not None and cell.piece is not None and cell.piece.owner != player_id:
                        sym = '?'  # Enemy pieces are '?' in the case player_id is not None
                    elif cell.piece is not None:
                        sym = cell.piece.rank
                cell_strs.append(sym.ljust(2))
            line = prefix + (' ' if prefix else '') + ' '.join(cell_strs)
            lines.append(line)

        # Add bottom player label
        if self.show_player_labels:
            lines.append(padding + f"Player {bottom_player}")
        return lines

    def _get_move_info_summaries(self, player_id: int | None = 0):
        """
        Generate textual summaries of the last two moves from the perspective of player_id.

        Args:
            player_id: Player ID (0 or 1) or None for global view, to orient cell labels correctly.

        Returns:
            A tuple (own_summary, opponent_summary), where each summary is a list of lines or None if unavailable.
        """
        # No moves yet: nothing to summarize
        if len(self.move_info_queue) == 0:
            return None, None

        # Only one move exists: determine whose move it is
        elif len(self.move_info_queue) == 1:
            if player_id in (0, None):
                # show own move for player 0 or global view
                opponent_move_info = None
                own_previous_info = self.move_info_queue[0]
            else:
                # show opponent's move for player 1 perspective
                opponent_move_info = self.move_info_queue[0]
                own_previous_info = None
        else:
            # Two or more moves: first is your own last, second is opponent last
            own_previous_info = self.move_info_queue[0]
            opponent_move_info = self.move_info_queue[1]

        # Convert raw move info dicts into human-readable string lists
        previous_move_summary = (
            None if own_previous_info is None
            else self._parse_move_info(own_previous_info, player_id)
        )
        opponent_move_summary = (
            None if opponent_move_info is None
            else self._parse_move_info(opponent_move_info, player_id)
        )
        return previous_move_summary, opponent_move_summary

    def _parse_move_info(self, move_info: MoveInfo, player_id: int | None = 0) -> list[str]:
        """
        Translate a single move_info dict into a list of narrative lines.

        Args:
            move_info: Dictionary containing keys 'player_id', 'src', 'dst',
                       'src_rank', 'dst_rank', 'winner', 'winning_rank'.
            player_id: Perspective for labeling cells and pronouns (0, 1, or None).

        Returns:
            A list of strings describing what happened on that turn, including any battle outcome.
        """
        summary: list[str] = [""]
        owner = move_info['player_id']
        # Convert indices back to cell labels (e.g., 'b3') for the given view
        src_cell = self._indices_to_cell(move_info['src'], player_id)
        dst_cell = self._indices_to_cell(move_info['dst'], player_id)

        # Movement without battle (destination was empty)
        if move_info['dst_rank'] is None:
            # If viewing your own piece, reveal rank; otherwise mask with '?'
            piece_label = move_info['src_rank'] if owner == player_id else '?'
            if player_id is None:
                summary.append(f"In the previous turn, Player {owner} moved his {piece_label} from {src_cell} to "
                               f"{dst_cell}. ")
            elif owner == player_id:
                summary.append(f"In your previous turn, you moved your {piece_label} from {src_cell} to {dst_cell}. ")
            else:
                summary.append(f"In his previous turn, your opponent moved his {piece_label} from {src_cell} to "
                               f"{dst_cell}. ")
        else:
            # There was a battle: include attacker and defender
            atk_rank = move_info['src_rank']
            def_rank = move_info['dst_rank']
            if player_id is None:
                summary.append(f"In the previous turn, Player {owner} moved his {atk_rank} from {src_cell} to "
                               f"{dst_cell}, attacking Player {1 - owner}'s {def_rank} in that cell. ")
            elif owner == player_id:
                summary.append(f"In your previous turn, you moved your {atk_rank} from {src_cell} to {dst_cell}, "
                               f"attacking the opponent's {def_rank} in that cell. ")
            else:
                summary.append(f"In his previous turn, your opponent moved his {atk_rank} from {src_cell} to "
                               f"{dst_cell}, attacking your {def_rank} in that cell. ")

            # Determine and narrate the battle result
            if move_info['winner'] is None:
                summary.append("Since both pieces are of the same rank, both were removed from the board. ")
            else:
                win_rank, lose_rank = move_info['winning_rank'], move_info['losing_rank']
                winner, loser = move_info['winner'], 1 - move_info['winner']

                add_reason = " (due to the aggressor advantage rule)" if (self.aggressor_advantage and win_rank ==
                                                                          lose_rank) else ""

                if player_id is None:
                    summary.append(f"Since Player {winner}'s {win_rank} beats Player {loser}'s {lose_rank}"
                                   f"{add_reason}, Player {loser}'s {lose_rank} was removed from the board. ")
                elif move_info['winner'] == player_id:
                    # You won the fight
                    summary.append(f"Since your {win_rank} beats your opponent's {lose_rank}{add_reason}, the "
                                   f"opponent's {lose_rank} was removed from the board. ")
                else:
                    # Opponent won
                    summary.append(f"Since your opponent's {win_rank} beats your {lose_rank}{add_reason}, your "
                                   f"{lose_rank} was removed from the board. ")

        # Always append the resulting board prompt
        summary.append("The resulting board from this turn was: ")
        summary.append("")
        return summary

    def whose_turn(self) -> int:
        """
        Returns the ID of the player whose turn it is (0 or 1).
        """
        return self._current_player

    def is_over(self) -> bool:
        """
        Determines if the game has ended:
        - A Flag has been captured, or
        - One player has no legal moves left.
        """
        # return True if a flag’s been captured or no movable pieces remain
        if self.captured is not None:
            return True

        player0_valid_moves = self._has_valid_moves(0)
        player1_valid_moves = self._has_valid_moves(1)
        return not (player0_valid_moves and player1_valid_moves)

    def scores(self) -> dict[int, float]:
        """
        Returns the payoff scores: winner 1.0, loser 0.0 if game over;
        otherwise {0: 0.0, 1: 0.0}.
        """
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
        Constructs a full text representation of the game state for the model.
        Includes rules summary (first view), instructions, previous moves,
        and current/previous board snapshots from the player's perspective.
        If player_id is None, returns a global state where both players' pieces are revealed.
        If player_id is not None, returns a state for the given player, with the opponent's pieces revealed.
        If at least one move has been made in the game, this will return the previous board that resulted after
        the current player's previous move, as well as the current board that resulted from the opponent's previous
        move.
        """
        lines: list[str] = ["Stratego — the goal is to capture the enemy Flag."]
        # Always print header/context

        if self._is_first_state(player_id):
            pass

        current_board = None
        if len(self.board_state_queues[player_id]) == 2:
            current_board = self.board_state_queues[player_id][1]
        previous_board = self.board_state_queues[player_id][0]

        previous_move_summary, opponent_move_summary = self._get_move_info_summaries(player_id)

        if previous_move_summary is not None:
            lines.extend(previous_move_summary)
        elif player_id == 0:
            lines.append("Here is the state of the board: \n")
        else:
            lines.append("Here is the previous state of the board: \n")

        lines.extend(previous_board)

        if opponent_move_summary is not None:
            lines.extend(opponent_move_summary)

        if current_board is not None:
            lines.extend(current_board)

        # Display a combined, chronological list of moves and removals
        if self.display_past_moves:
            # Build a unified list of (turn, description) tuples
            events: list[tuple[int, str]] = []

            # 1) Moves
            for mh in self.move_histories[0]:
                t = mh["turn"]
                mv = mh["move"]
                events.append((
                    t,
                    f"Player 0 played the move '{mv}' in turn {t}."
                ))
            for mh in self.move_histories[1]:
                t = mh["turn"]
                mv = mh["move"]
                events.append((
                    t,
                    f"Player 1 played the move '{mv}' in turn {t}."
                ))

            # 2) Removals
            for info in self.removed_pieces[0]:
                t = info["turn"]
                lost = info["lost_rank"] or "unknown"
                by = info["by_rank"] or "unknown"
                events.append((
                    t,
                    f"This move resulted in a battle; Piece {lost} of Player 0 was removed by Piece {by} of Player 1 "
                    f"in turn {t}."
                ))
            for info in self.removed_pieces[1]:
                t = info["turn"]
                lost = info["lost_rank"] or "unknown"
                by = info["by_rank"] or "unknown"
                events.append((
                    t,
                    f"This move resulted in a battle; Piece {lost} of Player 1 was removed by Piece {by} of Player 0 "
                    f"in turn {t}."
                ))

            # Only if we actually have any events, print the header and bullets
            if events:
                lines.append("")  # spacer
                lines.append("Here's a chronological list of all past moves and battle resolutions so far in the game:")

                # Sort by turn (moves first, then removals)
                events.sort(key=lambda e: (e[0], 0 if "played the move" in e[1] else 1))

                # Emit each event as a bullet
                for _, desc in events:
                    lines.append(f"- {desc}")

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

        Raises GameplayError on invalid moves or rule violations.
        Raises RuntimeError if a terminal move ends the game.
        """
        self.first_states[player_id] = False

        # Parse move string
        try:
            src_str, dst_str = move.split('-')
        except ValueError:
            raise GameplayError(f"Invalid move format: {move!r}. Use 'b2-c3'.")

        # Convert cell notation to internal board indices
        src_r, src_c = self._cell_to_indices(src_str, player_id)
        dst_r, dst_c = self._cell_to_indices(dst_str, player_id)

        src_cell = self.board[src_r][src_c]
        dst_cell = self.board[dst_r][dst_c]

        # Validate source cell
        if src_cell.piece is None:
            raise GameplayError(f"No piece at source {src_str!r}.")
        piece = src_cell.piece
        if piece.owner != player_id:
            raise GameplayError(f"Cannot move opponent's piece at {src_str!r}.")
        if piece.rank in ('B', 'F'):
            raise GameplayError(f"Cannot move immobile piece {piece.rank} at {src_str!r}.")

        # Validate destination cell
        if dst_cell.is_lake:
            raise GameplayError(f"Cannot move into lake at {dst_str!r}.")
        if dst_cell.piece and dst_cell.piece.owner == player_id:
            raise GameplayError(f"Cannot move onto your own piece at {dst_str!r}.")

        # Movement distance and path
        dr = dst_r - src_r
        dc = dst_c - src_c
        # Non-scout must move exactly one cardinal step
        if piece.rank != '2':
            if abs(dr) + abs(dc) != 1:
                raise GameplayError(f"Invalid move distance for piece {piece.rank} from {src_str!r} to {dst_str!r}.")
        else:
            # Scout moves must be straight-line and unobstructed
            if not (dr == 0 or dc == 0):
                raise GameplayError(f"Invalid Scout move: must be straight line from {src_str!r} to {dst_str!r}.")
            # Check step through path excluding src and dst:
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
            cr, cc = src_r + step_r, src_c + step_c
            while (cr, cc) != (dst_r, dst_c):
                cell_mid = self.board[cr][cc]
                if cell_mid.piece or cell_mid.is_lake:
                    cell_str = self._indices_to_cell((cr, cc), player_id=player_id)
                    raise GameplayError(f"Path blocked for Scout move at {cell_str}!")
                cr += step_r
                cc += step_c

        # Enforce bounce rule to prevent repeating the same two squares three times
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

        # Prepare a history record for this move
        move_info: MoveInfo = {
            'player_id': player_id,
            'src': (src_r, src_c),
            'dst': (dst_r, dst_c),
            'src_rank': src_cell.piece.rank,
            'dst_rank': None if dst_cell.piece is None else dst_cell.piece.rank,
            'winner': None,
            'winning_rank': None,
            'losing_rank': None
        }

        # Execute move or resolve battle
        if dst_cell.piece is None:
            # Simple relocation
            dst_cell.piece = piece
            src_cell.piece = None
        else:
            # Battle resolution based on rank interactions
            # 1) Flag capture: attacker wins and game ends
            if dst_cell.piece.rank == 'F':
                self.captured = dst_cell.piece
                dst_cell.piece = piece
                src_cell.piece = None

            # 2) Bomb encounter: only Miner ('3') defuses bomb
            elif dst_cell.piece.rank == 'B':
                if piece.rank == '3':
                    dst_cell.piece = piece
                    src_cell.piece = None
                else:
                    # Attacker loses and is removed
                    src_cell.piece = None

            # 3) Spy attack: spy kills only the Marshal ('10') when attacking
            elif piece.rank == 'S' and dst_cell.piece.rank == '10':
                dst_cell.piece = piece

            # 4) Numeric rank comparison: higher rank wins
            elif int(piece.numeric_rank) > int(dst_cell.piece.numeric_rank):
                dst_cell.piece = piece
                src_cell.piece = None
            elif int(piece.numeric_rank) < int(dst_cell.piece.numeric_rank):
                # Attacker loses
                src_cell.piece = None
            else:
                # Equal ranks, if aggressor_advantage game-mode is enabled, attacking piece wins, otherwise both pieces
                # lose and are removed
                if self.aggressor_advantage:
                    dst_cell.piece = piece
                    src_cell.piece = None
                else:
                    dst_cell.piece = None
                    src_cell.piece = None

            # Record battle winner if any piece remains
            if dst_cell.piece is not None:
                move_info['winner'] = player_id if dst_cell.piece.owner == player_id else 1 - player_id
                move_info['winning_rank'] = dst_cell.piece.rank

                atk_rank = move_info['src_rank']
                def_rank = move_info['dst_rank']

                # The attacker won
                if move_info['winner'] == player_id:
                    move_info['losing_rank'] = def_rank
                else:
                    # The defender won
                    move_info['losing_rank'] = atk_rank

                loser = 1 - cast(int, move_info['winner'])
                if move_info['losing_rank'] is not None:
                    removed_info = RemovedHistory(turn=self.turn_counter, lost_rank=move_info['losing_rank'],
                                                  by_rank=move_info['winning_rank'])
                    self.removed_pieces[loser].append(removed_info)
            else:
                # If no pieces remain, we need to record the loss for both players in removed_pieces
                # since this only happens for units of the same rank, we can append either src_rank or dst_rank
                removed_info = RemovedHistory(turn=self.turn_counter, lost_rank=move_info['src_rank'],
                                              by_rank=move_info['src_rank'])
                self.removed_pieces[0].append(removed_info)
                self.removed_pieces[1].append(removed_info)

        # Append move to history
        self.move_histories[player_id].append(MoveHistory(turn=self.turn_counter, move=move))

        # Maintain only last two move infos in info queue
        if len(self.move_info_queue) >= 2:
            self.move_info_queue = self.move_info_queue[-1:]
        self.move_info_queue.append(move_info)

        # Switch turn to the other player
        self._current_player = 1 - self._current_player
        self.turn_counter += 1

        # Update board snapshots for each perspective
        for pid in (0, 1, None):
            if len(self.board_state_queues[pid]) >= 2:
                self.board_state_queues[pid] = self.board_state_queues[pid][-1:]
            self.board_state_queues[pid].append(self._board_lines(pid, self.show_board_labels))
