import re
import os
import json  # Added for parsing JSON responses
from collections import Counter  # Added for validating piece counts
from typing import Dict, List, Optional, TextIO, Any
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
# Assuming classes.py exists in the same directory and defines Stratego and GameplayError
from classes import Stratego, GameplayError

# --- Constants ---
REQUIRED_PIECES = {'10': 1, '9': 1, '8': 2, '7': 3, '6': 4, '5': 4, '4': 4, '3': 5, '2': 8, 'S': 1, 'F': 1, 'B': 6}
EXPECTED_TOTAL_PIECES = sum(REQUIRED_PIECES.values())  # Should be 40
BOARD_SETUP_ROWS = 4
BOARD_SETUP_COLS = 10

# load your key & org from the environment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Prompts ---
SETUP_SYSTEM_PROMPT = f"""
You are an expert Stratego player assisting me in setting up the game board.
Your task is to generate a valid initial board setup for one player.

Rules for Setup:
- The setup occupies the {BOARD_SETUP_ROWS} rows closest to the player's side of the board, each with {BOARD_SETUP_COLS} columns.
- The following dictionary contains piece, count pairs, where the count specifies how many of that given piece are required. You must place exactly 'count' many of the corresponding 'piece', no more and no less: {REQUIRED_PIECES}.
- Total pieces must be exactly {EXPECTED_TOTAL_PIECES}.
- Represent pieces using their standard symbols: '10' (Marshal), '9' (General), '8' (Colonel), '7' (Major), '6' (Captain), '5' (Lieutenant), '4' (Sergeant), '3' (Miner), '2' (Scout), 'S' (Spy), 'B' (Bomb), 'F' (Flag).

Output Format:
Provide your response as a JSON object containing a single key "setup".
The value associated with "setup" must be a list of {BOARD_SETUP_ROWS} lists.
Each inner list must contain {BOARD_SETUP_COLS} strings, where each string is a valid piece symbol.
The first list will be placed on row 4 of the board, the second list on row 3 of the board, the third list on row 2 of the board and the fourth list on row 1 of the board (the bottom).

Example of desired JSON output format:
{{
  "setup": [
    ["2", "2", "7", "8", "6", "2", "9", "7", "2", "6"],
    ["3", "3", "10", "4", "8", "5", "S", "4", "7", "2"],
    ["B", "3", "4", "3", "2", "B", "5", "6", "5", "B"],
    ["F", "B", "3", "5", "B", "2", "4", "6", "B", "2"],
  ]
}}

Generate a strong, strategic setup. Ensure the piece counts and dimensions are exactly correct.
"""

GAMEPLAY_SYSTEM_PROMPT = (
    "You are an expert board game player assisting me in a game of Stratego. "
    "I'll give you the current state of the game and you will suggest my next move.\n\n"
    "Quick summary of the rules:\n"
    "• Units can move one tile in each of the cardinal directions. Scouts ('2') can move multiple tiles as long as "
    "there are no lakes or units in the intermediate tiles.\n"
    "• A piece cannot move back and forth between the same two squares in three consecutive turns.\n"
    "• Only one piece can be moved on a turn.\n"
    "• Moving into a cell containing an enemy unit attacks that unit.\n"
    "• Bombs ('B') and flags ('F') cannot be moved. Bombs beat all units except Miners ('3'), who defuse bombs.\n"
    "• Spies ('S') defeat the Marshal ('10') only when attacking; otherwise they lose to any attacker.\n"
    "• Higher rank wins (e.g. '7' beats '6'); on ties both pieces are removed (unless the aggressor‐advantage rule is "
    "enabled, in which case the attacking piece wins on a tie).\n"
    "• Capturing the opponent's flag ('F') wins the game. A player with no valid moves also loses.\n\n"
    "Cell notation and move format:\n"
    "Cells are labeled by file (a–j) and rank (1–10) and may contain:\n"
    "  '.' empty, 'L' lake, 'B' bomb, 'F' flag, numbers '2'–'10' for units, 'S' spy, '?' hidden enemy.\n"
    "Specify moves as 'b2-b3'. Scouts can move across clear paths (e.g., 'b2-b5') but cannot jump lakes or units.\n"
    "Moves must land on empty ('.') or enemy ('?') cells only.\n\n"
    "Provide a bit of reasoning for your next move, then enclose your final decision in valid JSON with the key "
    "\"move\" and a value of the move string."
)
explanation_assistant_moves = "Here are the previous moves you made or attempted to make: \n"


# --- Helper Functions ---

def validate_setup(setup: Any, player_id: int) -> Optional[str]:
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
        return f"Invalid setup for Player {player_id}: Expected {EXPECTED_TOTAL_PIECES} total pieces, got {len(all_pieces)}."

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
            return f"Invalid setup for Player {player_id}: Incorrect count for piece '{piece}'. Expected {required_count}, got {actual_count}."

    return None  # Validation passed


def generate_setup(
        player_id: int,
        client: OpenAI,
        f: TextIO,
        n_attempts_per_request: Optional[int] = 3,
        verbose: bool = False
) -> Optional[List[List[str]]]:
    """
    Generates and validates a Stratego setup for a given player using the AI.
    Logs prompts, responses, and status messages to file f, and prints them if verbose.
    """
    header = f"--- Generating setup for Player {player_id} ---\n"
    f.write(header)
    if verbose:
        print(header.strip())

    messages: List[Any] = [
        ChatCompletionSystemMessageParam(role="system", content=SETUP_SYSTEM_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=f"Generate a setup for Player {player_id}.")
    ]
    attempts = 0

    while True:
        if n_attempts_per_request is not None and attempts >= n_attempts_per_request:
            error_msg = f"Error: Failed to generate a valid setup for Player {player_id} after {attempts} attempts.\n"
            f.write(error_msg)
            print(error_msg.strip())
            return None

        attempts += 1
        if attempts > 1:
            info = f"Attempt {attempts} for Player {player_id} setup generation...\n"
            f.write(info)
            if verbose:
                print(info.strip())

        # Log and print the exact prompt
        if verbose:
            print(f"\n[Player {player_id} Setup Prompt - Attempt {attempts}]")
        f.write("PROMPT:\n")
        for msg in messages:
            line = f"{msg['role']}: {msg['content']}\n"
            f.write(line)
            if verbose:
                print(line.strip())
        if verbose:
            print("-" * 20)
        f.write("\n")

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            text = raw.strip()

            # Log and print the exact response
            f.write("RESPONSE:\n" + text + "\n\n")
            if verbose:
                print(f"\n[Player {player_id} Setup Response - Attempt {attempts}]\n{text}\n" + "-" * 20)

            # Parse and validate
            parsed_json = json.loads(text)
            if "setup" not in parsed_json:
                raise ValueError("Missing 'setup' key in JSON response.")
            generated_setup = parsed_json["setup"]
            validation_error = validate_setup(generated_setup, player_id)
            if validation_error:
                raise ValueError(validation_error)

            success_msg = f"Successfully generated and validated setup for Player {player_id}.\n\n"
            f.write(success_msg)
            if verbose:
                print(success_msg.strip())
            messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=text))
            return generated_setup

        except (json.JSONDecodeError, ValueError, AssertionError) as e:
            error_msg = (
                f"Invalid setup received (Attempt {attempts}): {e}. "
                "Please provide a valid setup in the specified JSON format with correct dimensions and piece counts.\n\n"
            )
            f.write(error_msg)
            if verbose:
                print(f"Validation/Parse Error for Player {player_id}: {error_msg.strip()}")
            messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=raw))
            messages.append(ChatCompletionUserMessageParam(role="user", content=error_msg))

        except Exception as e:
            error_msg = f"API Error during Player {player_id} setup generation (Attempt {attempts}): {e}\n\n"
            f.write(error_msg)
            print(error_msg.strip())
            # retry loop continues


def rollout_single(
        game_number: int,
        f: TextIO,
        setup_p0: List[List[str]],  # Added parameter
        setup_p1: List[List[str]],  # Added parameter
        n_moves_per_player: Optional[int] = None,
        n_attempts_per_turn: Optional[int] = None,
        verbose: bool = False
) -> None:
    """
    Play up to n_moves_per_player moves per player (2*n total), or until game over if None.
    For each turn, retry up to n_attempts_per_turn on invalid moves before aborting the game.
    Write each log entry immediately to file f and optionally to console.
    """
    chat_history: Dict[str, List[dict]] = {"0": [], "1": []}  # Removed "2" history key

    # Use the generated setups passed as arguments
    try:
        game = Stratego(setup_p0, setup_p1,
                        aggressor_advantage=True if game_number % 2 != 0 else False)  # Example: Odd games have advantage
        if verbose:
            print(f"Game {game_number} created. Aggressor advantage: {game.aggressor_advantage}")
    except Exception as e:
        error_msg = f"CRITICAL ERROR: Failed to initialize Stratego game {game_number} with provided setups. Error: {e}\n"
        f.write(error_msg)
        if verbose:
            print(error_msg)
        # Log the setups that caused the error
        f.write(f"Setup P0 causing error:\n{json.dumps(setup_p0, indent=2)}\n")
        f.write(f"Setup P1 causing error:\n{json.dumps(setup_p1, indent=2)}\n")
        return  # Cannot continue this game

    # Determine system prompt based on rules for this game
    if game.aggressor_advantage:
        system_prompt = GAMEPLAY_SYSTEM_PROMPT + (
            "\nSince the Aggressor Advantage rule is enabled, when two units with the "
            "same rank battle, the attacking piece wins. ")
    else:
        system_prompt = GAMEPLAY_SYSTEM_PROMPT + (
            "\nSince the Aggressor Advantage rule is disabled, when two units with the "
            "same rank battle, both are removed from the game. ")

    f.write(f"########## ROLLOUT {game_number} ##########\n")
    f.write(f"Aggressor Advantage: {game.aggressor_advantage}\n\n")
    f.write(f"Initial Setup Player 0:\n{json.dumps(setup_p0, indent=2)}\n\n")
    f.write(f"Initial Setup Player 1:\n{json.dumps(setup_p1, indent=2)}\n\n")

    if verbose:
        print(f"########## ROLLOUT {game_number} ##########")
        print(f"Aggressor Advantage: {game.aggressor_advantage}\n")
        # Optional: print setups to console if verbose
        # print(f"Initial Setup Player 0:\n{json.dumps(setup_p0, indent=2)}\n")
        # print(f"Initial Setup Player 1:\n{json.dumps(setup_p1, indent=2)}\n")

    iteration = 0
    max_iterations = (n_moves_per_player * 2) if n_moves_per_player is not None else None

    while not game.is_over():
        # stop if we've reached the maximum moves
        if max_iterations is not None and iteration >= max_iterations:
            break

        pid = game.whose_turn()
        header = (
                f"--- Game {game_number} | Player {pid}'s turn "
                f"(move {iteration // 2 + 1}.{pid + 1}"  # More conventional move numbering (e.g., 1.1, 1.2, 2.1, 2.2)
                + (
                    f" - total turn {iteration + 1}/{max_iterations}" if max_iterations is not None else f" - total turn {iteration + 1}")
                + ") ---\n"
        )
        f.write(header + "\n")
        if verbose:
            print(header.strip())

        attempts = 0
        while True:  # Retry loop for getting a valid move
            state_str = game.state(pid)

            # gather all prior assistant messages for this player
            assistant_history = [
                # Ensure correct type hinting for messages
                m for m in chat_history[str(pid)] if m["role"] == "assistant"
            ]
            user_history = [
                m for m in chat_history[str(pid)] if m["role"] == "user"
            ]

            # start building messages with the system prompt
            # Correct type hinting for the list
            current_turn_messages: List[
                ChatCompletionSystemMessageParam |
                ChatCompletionAssistantMessageParam |
                ChatCompletionUserMessageParam
                ] = [
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            ]

            # If there's history, structure it conversationally
            # Combine user and assistant messages in order
            full_history = sorted(chat_history[str(pid)],
                                  key=lambda x: x.get('timestamp', 0))  # Assuming you add timestamps later if needed

            if full_history:
                current_turn_messages.append(
                    ChatCompletionUserMessageParam(role="user", content=explanation_assistant_moves)
                )
                for msg in full_history:
                    if msg["role"] == "assistant":
                        current_turn_messages.append(
                            ChatCompletionAssistantMessageParam(role="assistant", content=msg["content"]))
                    elif msg["role"] == "user":  # Include previous user prompts (like error corrections)
                        current_turn_messages.append(
                            ChatCompletionUserMessageParam(role="user", content=msg["content"]))

            # finally append the current-state prompt
            current_turn_messages.append(
                ChatCompletionUserMessageParam(role="user",
                                               content="Current board state:\n" + state_str + "\nWhat is your next move?")
            )

            f.write("PROMPT: \n")
            if verbose:
                print("\nPROMPT: \n")

            # Log the messages being sent for this turn's API call
            for msg2 in current_turn_messages:
                log_line = f"{msg2['role']}: {msg2['content']}\n"
                f.write(log_line)
                if verbose:
                    print(log_line.strip())
            f.write("\n")
            if verbose:
                print()

            # call the API
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=current_turn_messages,
                    # response_format={"type": "json_object"}, # Can optionally enforce JSON here too
                )
                raw = response.choices[0].message.content
                assert raw is not None, "API returned no content for move generation"
                text = raw.strip()
            except Exception as e:
                error_entry = f"API Error during move generation: {e}. Retrying...\n\n"
                f.write(error_entry)
                if verbose:
                    print(error_entry.strip())
                # Optionally add a delay before retrying API call
                continue  # Retry the API call

            f.write("RESPONSE: \n" + text + "\n\n")
            if verbose:
                print("RESPONSE: \n" + text + "\n")

            # Store assistant response immediately in history for this player
            # Add a placeholder timestamp if needed later for sorting
            chat_history[str(pid)].append({"role": "assistant", "content": text, "timestamp": iteration})

            # extract move
            move = None
            # Try extracting from JSON first
            try:
                match_json = re.search(
                    r'\{.*?\"move\"\s*:\s*\"([a-j][1-9]|a10|b10|c10|d10|e10|f10|g10|h10|i10|j10)-([a-j][1-9]|a10|b10|c10|d10|e10|f10|g10|h10|i10|j10)\".*?\}',
                    text, re.DOTALL | re.IGNORECASE)
                if match_json:
                    json_part = match_json.group(0)
                    try:
                        move_data = json.loads(json_part)
                        move = move_data.get("move")
                    except json.JSONDecodeError:
                        pass  # Fallback to regex if JSON is malformed

                # Fallback to simple regex if JSON fails or isn't present
                if not move:
                    match_move = re.search(r'"move"\s*:\s*"([a-j]\d{1,2}-[a-j]\d{1,2})"', text, re.IGNORECASE) or \
                                 re.search(r'([a-j]\d{1,2}-[a-j]\d{1,2})', text)  # More lenient regex as fallback
                    if match_move:
                        move = match_move.group(1)

            except Exception as e:
                f.write(f"Error parsing move from response: {e}\nText: {text}\n\n")
                if verbose:
                    print(f"Error parsing move from response: {e}")

            if not move:
                attempts += 1
                error_msg = f"Could not extract a valid move string (e.g., 'a1-a2') from the response (Attempt {attempts})"
                if n_attempts_per_turn is not None:
                    error_msg += f"/{n_attempts_per_turn}"
                error_msg += "). Please provide the move in the specified JSON format."
                f.write(error_msg + "\nRetrying...\n\n")
                if verbose:
                    print(error_msg + "\nRetrying...")

                # Add user message asking for correction, store in history
                chat_history[str(pid)].append({
                    "role": "user",
                    "content": error_msg,
                    "timestamp": iteration + 0.5  # Ensure it comes after assistant msg
                })
                # Check attempts before continuing
                if n_attempts_per_turn is not None and attempts >= n_attempts_per_turn:
                    abort_msg = (
                        f"Aborting Game {game_number}: exceeded {n_attempts_per_turn} "
                        f"attempts to extract a valid move.\n\n"
                    )
                    f.write(abort_msg)
                    if verbose:
                        print(abort_msg.strip())
                    return
                continue  # Go back to API call within the move retry loop

            # try playing the extracted move
            try:
                action_entry = f"EXTRACTED ACTION: {move}\n\n"
                f.write(action_entry)
                if verbose:
                    print(action_entry.strip())
                game.play(move, pid)
                iteration += 1  # Increment *total* turns played
                break  # success: exit move retry loop (while True)

            except GameplayError as e:
                attempts += 1
                error_entry = f"Error: {e} (Attempt {attempts}"
                if n_attempts_per_turn is not None:
                    error_entry += f"/{n_attempts_per_turn}"
                error_entry += ")\nPlease choose a different, valid move.\n\n"
                f.write(error_entry)
                if verbose:
                    print(error_entry.strip())

                # record the retry prompt in history for the current player
                chat_history[str(pid)].append({
                    "role": "user",
                    "content": f"That move ({move}) was illegal: {e}. Please choose another valid move.",
                    "timestamp": iteration + 0.5  # Ensure user error msg comes after assistant response
                })

                # if we've exceeded allowed attempts for this *turn*
                if n_attempts_per_turn is not None and attempts >= n_attempts_per_turn:
                    abort_msg = (
                        f"Aborting Game {game_number}: exceeded {n_attempts_per_turn} "
                        f"invalid move attempts on a single turn.\n\n"
                    )
                    f.write(abort_msg)
                    if verbose:
                        print(abort_msg.strip())
                    return  # Exit rollout_single

                # No break here, continue the inner while loop to ask the AI again

        # end of retry loop for a single turn, continue to next player's turn (outer while loop)

    # end of game or move limit reached
    scores = game.scores()

    footer = (
            f"=== Game {game_number} stopped after {iteration} total turns "
            f"(Player 0: {iteration // 2 + iteration % 2} turns, Player 1: {iteration // 2} turns)"
            + (f" - Limit: {n_moves_per_player} moves per player)" if n_moves_per_player is not None else "")
            + f" ===\n"
    )
    footer += f"Final Scores: {scores}\n"

    # Log final board state
    footer += f"\nFinal Board State (Player 0 perspective):\n{game.state(0)}\n"
    footer += f"\nFinal Board State (Player 1 perspective):\n{game.state(1)}\n"


def main():
    # configure here
    n_moves_per_player = 5  # Max moves per player (e.g., 50), or None to play until game over
    n_attempts_setup = 10  # Max attempts to get a valid setup from AI
    n_attempts_per_turn = 3  # Max attempts for AI to provide a valid move, or None for unlimited retries
    n_games = 1  # Number of games to simulate
    verbose = True  # Print progress and prompts/responses to console

    # --- Setup OpenAI Client ---
    # (Moved client initialization outside the loop to reuse it)
    global client  # Make client global or pass it around if preferred

    if 'client' not in globals() or client is None:
        print("Failed to initialize OpenAI client in main.")
        return

    for i in range(1, n_games + 1):
        start_msg = f"\n----- Starting Game {i} -----\n"
        print(start_msg.strip())
        fname = f"rollout_{i}.log"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(start_msg)

            # --- Generate Setups ---
            print("Generating setup for Player 0...")
            setup_p0 = generate_setup(0, client, f, n_attempts_setup, verbose)
            if setup_p0 is None:
                abort_msg = f"ABORTED GAME {i}: Failed to generate setup for Player 0.\n"
                print(abort_msg.strip())
                f.write(abort_msg)
                continue

            print("Generating setup for Player 1...")
            setup_p1 = generate_setup(1, client, f, n_attempts_setup, verbose)
            if setup_p1 is None:
                abort_msg = f"ABORTED GAME {i}: Failed to generate setup for Player 1.\n"
                print(abort_msg.strip())
                f.write(abort_msg)
                continue

            # --- Run Game Simulation ---
            sim_msg = f"Starting simulation for Game {i}..."
            print(sim_msg)
            f.write(sim_msg + "\n")
            rollout_single(
                game_number=i,
                f=f,
                setup_p0=setup_p0,  # Pass generated setup
                setup_p1=setup_p1,  # Pass generated setup
                n_moves_per_player=n_moves_per_player,
                n_attempts_per_turn=n_attempts_per_turn,
                verbose=verbose
            )
        print(f"\nSaved rollout {i} to {fname}")
        print(f"----- Finished Game {i} -----")


if __name__ == "__main__":
    main()