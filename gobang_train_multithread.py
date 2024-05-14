import atexit
import queue
import threading
import time

from gobang_train import robot_step
import torch
import agent
import environment

TRAIN_TIME = 500
BOARD_SIZE = 8
WIN_SIZE = 5
MODULE_SAVE_PATH = "./best_gobang_multi.pth"
MODULE_UE_SAVE_PATH = "./best_gobang_multi_ue.pth"
LEARNING_RATE = 0.0001
SHOW_BOARD_TIME = 10
DEVICE = torch.device("cpu")  # if you wait to use cuda: "DEVICE = torch.device("cuda")"
MAX_MEMORY_SIZE = 51200
BATCH_SIZE = 10240
VALID_EPOCH = 5
VALID_GAME_NUMBERS = 10

# THREAD_NUM base on EPSILON_LIST
# each tuple in EPSILON_LIST is (ROBOT_A_EPSILON, ROBOT_A_EPSILON_DECAY, ROBOT_B_EPSILON, ROBOT_B_EPSILON_DECAY)
# when ROBOT_A_EPSILON==-1 means use gobang_dm
EPSILON_LIST = [
    (0.8, 0.99, 0, 1),
    (0.8, 0.99, 0.1, 1),
    (0.8, 0.99, 0.2, 1),
    (0.8, 0.99, 0.3, 1),
    (0.8, 0.99, 0.4, 1),
    (0.8, 0.99, 0.5, 1),
    (0.8, 0.99, 0.6, 1),
    (0.8, 0.99, 0.7, 1),
    (0.8, 0.99, 0.8, 1),
    (0.8, 0.99, 0.9, 1),
    (0.8, 0.99, 1, 1),
    (0.8, 0.99, 1, 1),
    (0.8, 0.99, -1, 1),
    (0.8, 0.99, -1, 1),
    (-1, 1, 0.8, 0.99),
    (-1, 1, 0.8, 0.99),
    (-1, 1, -1, 1),
    (-1, 1, -1, 1),
    (1, 1, -1, 1),
    (1, 1, -1, 1),
    (-1, 1, 1, 1),
    (-1, 1, 1, 1),
    (1, 1, 1, 1),
    (1, 1, 1, 1)
]

# epsilon and epsilon_decay are meaningless for the next robot
tol_robot = agent.gobang.robot(
    device=DEVICE,
    epsilon=0,
    epsilon_decay=1,
    board_size=BOARD_SIZE,
    lr=LEARNING_RATE,
    max_memory_size=MAX_MEMORY_SIZE,
    batch_size=BATCH_SIZE
)

thread_num = len(EPSILON_LIST)

# thread semaphore
game_over_count = threading.Semaphore(0)
start_next_game = threading.Semaphore(0)

game_info = queue.Queue()


def train(robot_a_episode: float, robot_a_episode_decay: float, robot_b_episode: float, robot_b_episode_decay: float):
    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)

    # board_size and lr are meaningless for the next two rebot.
    # set max_memory_size=0 is avoiding to train at these two robots.
    if robot_a_episode != -1:
        robot_A = agent.gobang.robot(
            device=DEVICE,
            epsilon=robot_a_episode,
            epsilon_decay=robot_a_episode_decay,
            board_size=BOARD_SIZE,
            lr=LEARNING_RATE,
            max_memory_size=0
        )
    else:
        robot_A = agent.gobang_dm.dm_robot(env=env, color=env.A)

    if robot_b_episode != -1:
        robot_B = agent.gobang.robot(
            device=DEVICE,
            epsilon=robot_b_episode,
            epsilon_decay=robot_b_episode_decay,
            board_size=BOARD_SIZE,
            lr=LEARNING_RATE,
            max_memory_size=0
        )
    else:
        robot_B = agent.gobang_dm.dm_robot(env=env, color=env.B)

    for epoch in range(TRAIN_TIME):
        start_next_game.acquire(blocking=True)

        # get module from tol_robot
        if robot_a_episode != -1:
            robot_A.change_module_from_other(tol_robot)
        if robot_b_episode != -1:
            robot_B.change_module_from_other(tol_robot)

        env.clear()
        cnt = 0
        while True:
            done = robot_step(env.A, robot_A, env, memorize_to_robot=tol_robot, is_train=False, board_size=BOARD_SIZE)
            cnt += 1
            if done != 0:
                who_win = done
                break

            done = robot_step(env.B, robot_B, env, memorize_to_robot=tol_robot, is_train=False, board_size=BOARD_SIZE)
            cnt += 1
            if done != 0:
                who_win = done
                break

        # game_info.put((cnt, who_win))

        robot_A.reduce_epsilon()
        robot_B.reduce_epsilon()

        game_over_count.release(1)


def valid(robot, valid_num: int = 10):
    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)
    robot_A = agent.gobang_dm.dm_robot(env=env, color=env.A)
    robot_B = agent.gobang_dm.dm_robot(env=env, color=env.B)

    A_draw_cnt, A_win_cnt, A_lose_cnt, A_tol_place = 0, 0, 0, 0
    B_draw_cnt, B_win_cnt, B_lose_cnt, B_tol_place = 0, 0, 0, 0

    with torch.no_grad():
        for _ in range(valid_num):
            while True:
                done = robot_step(env.A, robot, env, memorize_to_robot=None, is_train=False, board_size=BOARD_SIZE)
                A_tol_place += 1
                if done != 0:
                    who_win = done
                    break

                done = robot_step(env.B, robot_B, env, memorize_to_robot=None, is_train=False, board_size=BOARD_SIZE)
                A_tol_place += 1
                if done != 0:
                    who_win = done
                    break

            if who_win == env.draw_play:
                A_draw_cnt += 1
            elif who_win == env.A:
                A_win_cnt += 1
            else:
                A_lose_cnt += 1

            env.clear()

            while True:
                done = robot_step(env.A, robot_A, env, memorize_to_robot=None, is_train=False, board_size=BOARD_SIZE)
                B_tol_place += 1
                if done != 0:
                    who_win = done
                    break

                done = robot_step(env.B, robot, env, memorize_to_robot=None, is_train=False, board_size=BOARD_SIZE)
                B_tol_place += 1
                if done != 0:
                    who_win = done
                    break
            if who_win == env.draw_play:
                B_draw_cnt += 1
            elif who_win == env.B:
                B_win_cnt += 1
            else:
                B_lose_cnt += 1

            env.clear()

    print("\tdraw: {}, win: {}, loss: {} as playerA, using {:.3f} avg place.".format(A_draw_cnt, A_win_cnt, A_lose_cnt,
                                                                                     A_tol_place / valid_num))
    print("\tdraw: {}, win: {}, loss: {} as playerB, using {:.3f} avg place.".format(B_draw_cnt, B_win_cnt, B_lose_cnt,
                                                                                     B_tol_place / valid_num))


def main():
    @atexit.register
    def when_unexpect_exit():
        torch.save(tol_robot.module, MODULE_UE_SAVE_PATH)
        print("[note] because unexpected exit, we save current net as '{}'.".format(MODULE_UE_SAVE_PATH))

    for args in EPSILON_LIST:
        sub = threading.Thread(target=train, args=args)
        sub.daemon = True
        sub.start()

    avg_time = 0
    for epoch in range(TRAIN_TIME):
        start_time = time.time()

        # let the games begin
        start_next_game.release(thread_num)

        # wait for all games over
        for _ in range(thread_num):
            game_over_count.acquire(blocking=True)

        # train
        tol_robot.train_memory()

        diff_time = time.time() - start_time

        avg_time = 0.5 * (avg_time + diff_time)
        print(f"Epoch {epoch + 1}/{TRAIN_TIME}, {diff_time:.3f}it/s, {avg_time * (TRAIN_TIME - epoch - 1):.0f}s left.")

        if epoch % VALID_EPOCH == 0:
            valid(tol_robot, valid_num=VALID_GAME_NUMBERS)
            print("\tState count: {}".format(len(tol_robot.memory)))

    tol_robot.save(MODULE_SAVE_PATH)

    atexit.unregister(when_unexpect_exit)


if __name__ == '__main__':
    main()
