import atexit

import torch
import time
import agent
import environment

TRAIN_TIME = 20000
BOARD_SIZE = 3
WIN_SIZE = 3
MODULE_SAVE_PATH = "./best_gobang.pth"
MODULE_UE_SAVE_PATH = "best_gobang_ue.pth"
LEARNING_RATE = 0.0001
SHOW_BOARD_TIME = 10
DEVICE = torch.device("cpu")  # if you wait to use cuda: "DEVICE = torch.device("cuda")"

ROBOT_A_EPSILON = 0.8
ROBOT_A_EPSILON_DECAY = 0.99
ROBOT_B_EPSILON = 0.3
ROBOT_B_EPSILON_DECAY = 1


def robot_step(who, robot, env, memorize_to_robot=None, is_train: bool = True, show_result: bool = False):
    state = env.get_state(who)
    action = robot.get_action(state)

    place_hash = torch.argmax(action).item()
    if show_result:
        print(f"chosen:{place_hash}:{action.detach().cpu().numpy()}")
    r, c = place_hash // BOARD_SIZE, place_hash % BOARD_SIZE
    env.step(who, (r, c))

    done = env.check()
    reward = env.get_reward()
    if show_result:
        print(f"reward:{reward}")
    next_state = env.get_state(env.B if who == env.A else env.A)

    action = action.cpu().detach().numpy()

    if memorize_to_robot is not None:
        memorize_to_robot.memorize(state, action, reward, next_state, done)
    if is_train:
        robot.train_action(state, action, reward, next_state, done)

    return done


def train():
    robot = agent.gobang.robot(
        device=DEVICE,
        epsilon=ROBOT_A_EPSILON,
        epsilon_decay=ROBOT_A_EPSILON_DECAY,
        board_size=BOARD_SIZE,
        lr=LEARNING_RATE
    )
    robot_best = agent.gobang.robot(
        device=DEVICE,
        epsilon=ROBOT_B_EPSILON,
        epsilon_decay=ROBOT_B_EPSILON_DECAY,
        board_size=BOARD_SIZE,
        lr=LEARNING_RATE
    )

    @atexit.register
    def when_unexpect_exit():
        if robot:
            torch.save(robot.module, MODULE_UE_SAVE_PATH)
            print("[note] because unexpected exit, we save current net as '{}'.".format(MODULE_UE_SAVE_PATH))

    robot.save(MODULE_SAVE_PATH)

    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)

    avg_time = 0
    for epoch in range(TRAIN_TIME):
        start_time = time.time()
        env.clear()
        while True:
            if epoch % SHOW_BOARD_TIME == 0:
                print("A:", robot.module(
                    torch.tensor(env.get_state(env.A)).float().unsqueeze(0).to(DEVICE)).cpu().detach().numpy())

            # player1(train)
            done = robot_step(env.A, robot, env, memorize_to_robot=robot, is_train=True)

            if done != 0:
                who_win = done
                break

            # player2(best)
            if epoch % SHOW_BOARD_TIME == 0:
                print("B:", robot.module(
                    torch.tensor(env.get_state(env.B)).float().unsqueeze(0).to(DEVICE)).cpu().detach().numpy())

            done = robot_step(env.B, robot_best, env, memorize_to_robot=robot, is_train=False)

            if done != 0:
                who_win = done
                break

        robot.train_memory()

        diff_time = time.time() - start_time
        avg_time = 0.5 * (avg_time + diff_time)
        print(f"Epoch {epoch + 1}/{TRAIN_TIME}, {diff_time:.3f}it/s, {avg_time * (TRAIN_TIME - epoch - 1):.0f}s left:")

        if epoch % SHOW_BOARD_TIME == 0:
            env.display()

        if who_win == env.draw_play:
            print(f"draw after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")
            continue
        if who_win == env.A:
            print(f"Player1 win after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")
            robot_best.change_module_from_other(robot)
        if who_win == env.B:
            if epoch % SHOW_BOARD_TIME:
                env.display()
            print(f"Player2 win after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")

        robot.reduce_epsilon()
        robot_best.reduce_epsilon()

    robot.save(MODULE_SAVE_PATH)
    atexit.unregister(when_unexpect_exit)


def play():
    robot = agent.gobang.robot(device=torch.device('cpu'), epsilon=0, board_size=BOARD_SIZE,
                               module_save_path=MODULE_SAVE_PATH)

    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)
    with torch.no_grad():
        while True:
            done = robot_step(env.A, robot, env, is_train=False, show_result=True)

            if done != 0:
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
    play()
