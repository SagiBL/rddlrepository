from mcts import MCTS


def play(state):
    mcts = MCTS(state)

    while not state.game_over():

        mcts.move(user_move)

        if state.game_over():
            print("Player one won!")
            break

        print("Thinking...")

        mcts.search(8)
        num_rollouts, run_time = mcts.statistics()
        print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
        move = mcts.best_move()

        print("MCTS chose move: ", move)

        state.move(move)
        mcts.move(move)

        if state.game_over():
            print("Player two won!")
            break


if __name__ == "__main__":
    play()
