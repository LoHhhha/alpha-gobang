import atexit
import datetime
import queue
import threading
import time

from gobang_train import robot_step
import torch
import agent
import environment

TRAIN_TIME = 2000
BOARD_SIZE = 8
WIN_SIZE = 5
LEARNING_RATE = 0.01
DEVICE = torch.device("cpu")  # if you wait to use cuda: "DEVICE = torch.device("cuda")"
MAX_MEMORY_SIZE = 2560
BATCH_SIZE = 256
VALID_EPOCH = 5
VALID_GAME_NUMBERS = 10

STATR_TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
MODULE_SAVE_PATH = (f"./alpha_gobang_B{BOARD_SIZE}_W{WIN_SIZE}_"
                    f"{STATR_TIME}_multi.pth")
BEST_MODULE_SAVE_PATH = (f"./alpha_gobang_B{BOARD_SIZE}_W{WIN_SIZE}_"
                         f"{STATR_TIME}_multi_best.pth")
MODULE_UE_SAVE_PATH = (f"./alpha_gobang_B{BOARD_SIZE}_W{WIN_SIZE}_"
                       f"{STATR_TIME}_multi_ue.pth")
BEST_UE_MODULE_SAVE_PATH = (f"./alpha_gobang_B{BOARD_SIZE}_W{WIN_SIZE}_"
                            f"{STATR_TIME}_multi_best_ue.pth")

# THREAD_NUM base on EPSILON_LIST
# each tuple in EPSILON_LIST is (ROBOT_A_EPSILON, ROBOT_A_EPSILON_DECAY, ROBOT_B_EPSILON, ROBOT_B_EPSILON_DECAY)
# when ROBOT_A_EPSILON==-1 means use gobang_dm
EPSILON_LIST = [
    (0, 1, 0, 1),
    (0, 1, 0, 1),
    (0, 1, 0, 1),
    (0, 1, 0, 1),
    (0, 1, -1, 1),
    (0, 1, -1, 1),
    (0, 1, -1, 1),
    (0, 1, -1, 1),
    (-1, 1, 0, 1),
    (-1, 1, 0, 1),
    (-1, 1, 0, 1),
    (-1, 1, 0, 1),
    (-1, 1, -1, 1),
    (-1, 1, -1, 1),
    (-1, 1, -1, 1),
    (-1, 1, -1, 1),
    (1, 1, 1, 1),
    (1, 1, 1, 1),
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

torch.manual_seed(19528)

best_module = tol_robot.module
best_score = -1

thread_num = len(EPSILON_LIST)

# thread semaphore
game_over_count = threading.Semaphore(0)
start_next_game = threading.Semaphore(0)

game_info = queue.Queue()


@torch.no_grad()
def view(robot_a_episode: float, robot_a_episode_decay: float, robot_b_episode: float, robot_b_episode_decay: float):
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

    global best_score, best_module
    score = A_draw_cnt + B_draw_cnt + A_win_cnt * 2 + B_win_cnt * 3
    if score >= best_score:
        best_score = score
        best_module.load_state_dict(robot.module.state_dict())


def main():
    @atexit.register
    def when_unexpect_exit():
        torch.save(tol_robot.module, MODULE_UE_SAVE_PATH)
        torch.save(best_module, BEST_UE_MODULE_SAVE_PATH)
        print("[note] because unexpected exit, we save current net as '{}'.".format(MODULE_UE_SAVE_PATH))
        print("[note] because unexpected exit, we save current net as '{}'.".format(BEST_UE_MODULE_SAVE_PATH))

    for args in EPSILON_LIST:
        sub = threading.Thread(target=view, args=args)
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
    best_module.save(BEST_MODULE_SAVE_PATH)

    atexit.unregister(when_unexpect_exit)


if __name__ == '__main__':
    main()
