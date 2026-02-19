
def convert_to_board(word, n):
    """Docstring

    """
    word = word.astype(np.float32)  # cast input array to float32
    board = np.zeros((n, n), dtype=np.float32)
    for i in range(len(word)):
        board[i//n, i%n] = word[i]
    return board


# Defining a helper function that takes in a game and outputs the final board state
def final_board_state(game):
    """Docstring

    """
    n = len(game)
    for i in range(len(game)):
        if i == 0:
            continue
        if i == n-1:
            return game[i]
        if game[i+1].sum() == 0 and game[i].sum() != 0:
            return game[i]
