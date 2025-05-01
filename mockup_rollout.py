import re
from typing import Dict, List
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from classes import Stratego, GameplayError


setup_p0 = [
    # front-line (row 4 of the board)
    ['2', '2', '7', '8', '6', '2', '9', '7', '2', '6'],

    # row 3
    ['3', '3', '10', '4', '8', '5', 'S', '4', '7', '2'],

    # row 2
    ['B', '3', '4', '3', '2', 'B', '5', '6', '5', 'B'],

    # back-line (row 1)   ← Flag in the left corner
    ['F', 'B', '3', '5', 'B', '2', '4', '6', 'B', '2'],
]
setup_p1 = [
    # front-line (row 4 of the board)
    ['2', '7', '8', '6', '4', '3', 'S', '6', '2', '2'],

    # row 3
    ['2', '3', '5', '7', '9', '8', '5', '4', '6', '2'],

    # row 2
    ['B', '3', '4', '2', 'B', 'B', '5', '2', '7', '3'],

    # back-line (row 1)   ← Flag in the centre, ringed by bombs
    ['6', '5', '4', 'B', 'F', 'B', '10', '3', 'B', '2'],
]

SYSTEM_PROMPT = (
    "You are an expert board game player assisting me in a game of Stratego. "
    "I'll give you the current state of the game and you will suggest my next move.\n"
    "Provide some short reasoning about your next move, then enclose your final decision "
    "in valid JSON with the key \"move\" and a value with the move you wish to take. "
    "Make sure that you specify a valid move string (e.g. 'b2-c3')."
)


def rollout_single(game_number: int) -> List[str]:
    # history of messages per player, plus game-log lines
    chat_history: Dict[str, List[dict]] = {"0": [], "1": [], "2": []}
    game_log: List[str] = []

    game = Stratego(setup_p0, setup_p1)
    moves = ["e4-e5", "e4-e5", "e5-e6", "e5-e6", "e6-e7", "e6-e7"]
    i = 0

    while not game.is_over():
        pid = game.whose_turn()
        state_str = game.state(pid)
        header = f"--- Game {game_number} | Player {pid}'s turn ---\n{state_str}\n"

        # build a *typed* list of messages
        messages: List[ChatCompletionSystemMessageParam | ChatCompletionAssistantMessageParam
                       | ChatCompletionUserMessageParam] = [ChatCompletionSystemMessageParam(role="system",
                                                                                             content=SYSTEM_PROMPT)]

        # replay prior assistant messages
        messages += [ChatCompletionAssistantMessageParam(role=m["role"], content=m["content"])
                     for m in chat_history[str(pid)]]

        # # Replay every turn (both user and assistant) in the order they happened
        # for turn in chat_history[str(pid)]:
        #     if turn["role"] == "user":
        #         messages.append(ChatCompletionUserMessageParam(
        #             role="user",
        #             content=turn["content"]
        #         ))
        #     else:  # "assistant"
        #         messages.append(ChatCompletionAssistantMessageParam(
        #             role="assistant",
        #             content=turn["content"]
        #         ))

        # append the new user prompt
        messages.append(ChatCompletionUserMessageParam(role="user", content=state_str))

        # make the sync call (no await)
        text = f'{{"move": "{moves[i]}"}}'
        i += 1

        # record assistant reply
        # chat_history[str(pid)].append({"role": "user", "content": your_state_prompt})
        chat_history[str(pid)].append({"role": "assistant", "content": text})
        game_log.append(header + "Response: " + text + "\n")

        # parse the move
        m = re.search(r'"move"\s*:\s*"([^"]+)"', text)
        if not m:
            raise ValueError("No valid action found in response: " + text)
        move = m.group(1)

        # attempt it in the game
        try:
            game.play(move, pid)
            game_log.append(f"Action: {move}\n\n")
        except GameplayError as e:
            game_log.append(f"Error: {e}\nRetrying...\n\n")
            chat_history[str(pid)].append({
                "role": "user",
                "content": f"Illegal move: {e}. Please choose another move."
            })
            continue

        if i == 6:
            break

    # game over
    scores = game.scores()
    game_log.append(f"=== Game {game_number} Over ===\nScores: {scores}\n")
    return game_log


def main() -> None:
    all_logs: Dict[int, List[str]] = {}
    for i in range(1, 4):
        logs = rollout_single(i)
        all_logs[i] = logs

    # print + save each log
    for i, log in all_logs.items():
        print(f"\n########## ROLLOUT {i} ##########\n")
        for entry in log:
            print(entry)

        fname = f"rollout_{i}.log"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"########## ROLLOUT {i} ##########\n\n")
            for entry in log:
                f.write(entry + "\n")
        print(f"Saved rollout {i} to {fname}")


if __name__ == "__main__":
    main()
