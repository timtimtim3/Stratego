import re
import os
from typing import Dict, List, Optional, TextIO
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from classes import Stratego, GameplayError

# load your key & org from the environment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

setup_p0 = [
    ['2', '2', '7', '8', '6', '2', '9', '7', '2', '6'],
    ['3', '3', '10', '4', '8', '5', 'S', '4', '7', '2'],
    ['B', '3', '4', '3', '2', 'B', '5', '6', '5', 'B'],
    ['F', 'B', '3', '5', 'B', '2', '4', '6', 'B', '2'],
]
setup_p1 = [
    ['2', '7', '8', '6', '4', '3', 'S', '6', '2', '2'],
    ['2', '3', '5', '7', '9', '8', '5', '4', '6', '2'],
    ['B', '3', '4', '2', 'B', 'B', '5', '2', '7', '3'],
    ['6', '5', '4', 'B', 'F', 'B', '10', '3', 'B', '2'],
]

SYSTEM_PROMPT = (
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


def rollout_single(
    game_number: int,
    f: TextIO,
    n_moves_per_player: Optional[int] = None,
    n_attempts_per_turn: Optional[int] = None,
    verbose: bool = False
) -> None:
    """
    Play up to n_moves_per_player moves per player (2*n total), or until game over if None.
    For each turn, retry up to n_attempts_per_turn on invalid moves before aborting the game.
    Write each log entry immediately to file f and optionally to console.
    """
    chat_history: Dict[str, List[dict]] = {"0": [], "1": [], "2": []}
    game = Stratego(setup_p0, setup_p1, aggressor_advantage=True if game_number == 3 else False)

    if game.aggressor_advantage:
        system_prompt = SYSTEM_PROMPT + ("\nSince the Aggressor Advantage rule is enabled, when two units with the "
                                         "same rank battle, the attacking piece wins. ")
    else:
        system_prompt = SYSTEM_PROMPT + ("\nSince the Aggressor Advantage rule is disabled, when two units with the "
                                         "same rank battle, both are removed from the game. ")

    f.write(f"########## ROLLOUT {game_number} ##########\n\n")
    if verbose:
        print(f"########## ROLLOUT {game_number} ##########\n")

    iteration = 0
    max_iterations = (n_moves_per_player * 2) if n_moves_per_player is not None else None

    while not game.is_over():
        # stop if we've reached the maximum moves
        if max_iterations is not None and iteration >= max_iterations:
            break

        pid = game.whose_turn()
        header = (
            f"--- Game {game_number} | Player {pid}'s turn "
            f"(move {iteration+1}"
            + (f"/{max_iterations}" if max_iterations is not None else "")
            + ") ---\n"
        )
        f.write(header + "\n")
        if verbose:
            print(header.strip())

        attempts = 0
        while True:
            state_str = game.state(pid)

            # gather all prior assistant messages for this player
            assistant_history = [
                ChatCompletionAssistantMessageParam(role=m["role"], content=m["content"])
                for m in chat_history[str(pid)]
            ]

            # start building base_messages with the system prompt
            base_messages: List[
                ChatCompletionSystemMessageParam
                | ChatCompletionAssistantMessageParam
                | ChatCompletionUserMessageParam
                ] = [
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            ]

            # if they *have* made any moves before, show them:
            if assistant_history:
                base_messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=explanation_assistant_moves
                    )
                )

            # now replay their prior assistant turns
            base_messages += assistant_history

            # finally append the current-state prompt
            base_messages.append(
                ChatCompletionUserMessageParam(role="user", content=state_str)
            )

            f.write("PROMPT: \n")
            if verbose:
                print("PROMPT: \n")

            # First, log the system prompt itself:
            f.write(system_prompt + "\n\n")
            if verbose:
                print(system_prompt)
                print()

            if assistant_history:
                f.write(explanation_assistant_moves + "\n")
                if verbose:
                    print(explanation_assistant_moves + "\n")

            for m in chat_history[str(pid)]:
                f.write(f"{m["role"]}: {m["content"]}\n")
                if verbose:
                    print(f"{m["role"]}: {m["content"]}\n")

            f.write(state_str + "\n")
            if verbose:
                print(state_str + "\n")

            # call the API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=base_messages,
            )
            raw = response.choices[0].message.content
            assert raw is not None, "API returned no content"
            text = raw.strip()

            f.write("RESPONSE: \n" + text + "\n\n")
            if verbose:
                print("RESPONSE: \n" + text)

            chat_history[str(pid)].append({"role": "assistant", "content": text})

            # extract move
            match_move = re.search(r'"move"\s*:\s*"([^"]+)"', text)
            if not match_move:
                raise ValueError("No valid action found in response: " + text)
            move = match_move.group(1)

            # try playing
            try:
                action_entry = f"EXTRACTED ACTION: {move}\n\n"
                f.write(action_entry)
                if verbose:
                    print(action_entry.strip())
                game.play(move, pid)
                iteration += 1
                break  # success: exit retry loop
            except GameplayError as e:
                attempts += 1
                error_entry = f"Error: {e} (attempt {attempts}"
                if n_attempts_per_turn is not None:
                    error_entry += f"/{n_attempts_per_turn}"
                error_entry += ")\nRetrying...\n\n"
                f.write(error_entry)
                if verbose:
                    print(error_entry.strip())

                # record the retry prompt
                chat_history[str(pid)].append({
                    "role": "user",
                    "content": f"Illegal move: {e}. Please choose another move. \n"
                })

                # if we've exceeded allowed attempts
                if n_attempts_per_turn is not None and attempts >= n_attempts_per_turn:
                    abort_msg = (
                        f"Aborting Game {game_number}: exceeded {n_attempts_per_turn} "
                        f"invalid attempts on a single turn.\n\n"
                    )
                    f.write(abort_msg)
                    if verbose:
                        print(abort_msg.strip())
                    return

        # end of retry loop, continue to next turn

    # end of game or move limit
    scores = game.scores()
    footer = (
        f"=== Game {game_number} stopped after {iteration} moves"
        + (f" (each player max {n_moves_per_player})" if n_moves_per_player is not None else "")
        + f" ===\nScores: {scores}\n"
    )
    f.write(footer + "\n")
    if verbose:
        print(footer.strip())


def main():
    # configure here
    n_moves_per_player = 5   # or None to play until game over
    n_attempts_per_turn = 3  # or None for unlimited retries
    n_games = 3
    verbose = True

    for i in range(1, n_games + 1):
        fname = f"rollout_{i}.log"
        with open(fname, "w", encoding="utf-8") as f:
            rollout_single(
                game_number=i,
                f=f,
                n_moves_per_player=n_moves_per_player,
                n_attempts_per_turn=n_attempts_per_turn,
                verbose=verbose
            )
        print(f"Saved rollout {i} to {fname}\n")


if __name__ == "__main__":
    main()
