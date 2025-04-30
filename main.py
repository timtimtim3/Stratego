from classes import Stratego


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

game = Stratego(setup_p0, setup_p1, aggressor_advantage=True)

while True:
    player_id = game.whose_turn()
    state = game.state(player_id)
    print(state)
    move = input("Move for Player {}: \n".format(player_id))
    game.play(move, player_id)
