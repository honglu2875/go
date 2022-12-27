import random
from go_rl import FastState, FastBoard


def test_random_play():
    size = 19

    b = FastState()
    bb = FastBoard()
    b.init_game(size, 7.5)
    bb.reset_board(19)

    player = 0  # 0 for black
    for i in range(400):  # Limit to 400 steps
        search_count = 0
        valid, vertex = False, bb.PASS
        while True:
            search_count += 1
            if search_count >= 1000:
                break
            coord = (random.randint(0, size-1), random.randint(0, size-1))
            vertex = bb.get_vertex(*coord)  # 0-based, (column, row)
            valid = b.is_move_legal(player, vertex)
            if valid:
                break

        if not valid:
            print(f"Cannot find legal moves at step {i}")
            b.play_move(bb.PASS)
        else:
            b.play_move(vertex)
        player = 1 - player

    b.display_state()
    print(b.final_score())
