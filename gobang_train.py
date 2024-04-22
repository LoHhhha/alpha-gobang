import torch
import time
import agent
import environment

TRAIN_TIME = 1000
BOARD_SIZE = 8
WIN_SIZE = 5
MAX_FAIL_TIME = 10
MODULE_SAVE_PATH = "./best_gobang.pth"


def train():
    robot = agent.gobang.robot(device=torch.device('cuda'), epsilon=0.5, board_size=BOARD_SIZE)
    robot_best = agent.gobang.robot(device=torch.device('cuda'), epsilon=0.9, board_size=BOARD_SIZE)

    robot.save(MODULE_SAVE_PATH)

    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)

    avg_time = 0
    for epoch in range(TRAIN_TIME):
        start_time = time.time()
        env.clear()
        while True:
            # player1(train)
            fail_cnt = 0
            while True:
                state = env.get_state(1)
                train_action = robot.get_action(state, need_random=fail_cnt >= MAX_FAIL_TIME)

                place_hash = torch.argmax(train_action).item()
                r, c = place_hash // BOARD_SIZE, place_hash % BOARD_SIZE
                env.step(env.A, (r, c))

                done = env.check()
                reward = env.get_reward()
                next_state = env.get_state(env.A)

                train_action = train_action.cpu().detach().numpy()

                robot.memorize(state, train_action, reward, next_state, done)
                robot.train_action(state, train_action, reward, next_state, done)

                if env.pre_action is not None:
                    break

                fail_cnt += 1

            if done != 0:
                who_win = done
                break

            # player2(best)
            fail_cnt = 0
            while True:
                state = env.get_state(-1)
                best_action = robot_best.get_action(state, need_random=fail_cnt >= MAX_FAIL_TIME)

                place_hash = torch.argmax(best_action).item()
                r, c = place_hash // BOARD_SIZE, place_hash % BOARD_SIZE
                env.step(env.B, (r, c))

                done = env.check()
                reward = env.get_reward()
                next_state = env.get_state(env.B)

                robot_best.memorize(state, train_action, reward, next_state, done)
                robot_best.train_action(state, train_action, reward, next_state, done)

                if reward > 0:
                    robot_best.memorize(state, train_action, reward, next_state, done)

                if env.pre_action is not None:
                    break

                fail_cnt += 1

            if done != 0:
                who_win = done
                break

        robot.train_memory()

        diff_time = time.time() - start_time
        avg_time = 0.5 * (avg_time + diff_time)
        print(f"Epoch {epoch + 1}/{TRAIN_TIME}, {diff_time:.3f}it/s, {avg_time * (TRAIN_TIME - epoch - 1):.0f}s left:")

        if epoch % 10 == 0:
            env.display()

        if who_win == env.draw_play:
            print(f"draw after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")
            continue
        if who_win == 1:
            print(f"Player1 win after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")
            robot.save(MODULE_SAVE_PATH)
        if who_win == -1:
            print(f"Player2 win after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")
            robot_best.save(MODULE_SAVE_PATH)

        robot.change_module(MODULE_SAVE_PATH)
        robot_best.change_module(MODULE_SAVE_PATH)
        robot.reduce_epsilon()
        robot_best.reduce_epsilon()


def play():
    robot = agent.gobang.robot(device=torch.device('cuda'), epsilon=0, board_size=BOARD_SIZE,
                               module_save_path=MODULE_SAVE_PATH)

    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)
    with torch.no_grad():
        while True:
            need_random = False
            while True:
                if need_random:
                    print("[note] random chose!")

                state = env.get_state(env.A)
                best_action = robot.get_action(state, need_random)

                place_hash = torch.argmax(best_action).item()
                r, c = place_hash // BOARD_SIZE, place_hash % BOARD_SIZE
                env.step(env.A, (r, c))
                print((r, c), place_hash)

                need_random = True

                if env.pre_action is not None:
                    break

            env.display()

            while True:
                a = int(input("r->"))
                b = int(input("c->"))
                if env.board[a][b] != 0:
                    continue

                env.step(env.B, (a, b))

                if env.pre_action is not None:
                    break

            if env.check() != 0:
                break

        env.display()


if __name__ == '__main__':
    train()
