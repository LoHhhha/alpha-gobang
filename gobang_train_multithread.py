import atexit
import queue
import threading
import time

from gobang_train import robot_step
import torch
import agent
import environment

TRAIN_TIME = 20000
BOARD_SIZE = 3
WIN_SIZE = 3
MODULE_SAVE_PATH = "./best_gobang_multi.pth"
MODULE_UE_SAVE_PATH = "./best_gobang_multi_ue.pth"
LEARNING_RATE = 0.0001
SHOW_BOARD_TIME = 10
DEVICE = torch.device("cpu")  # if you wait to use cuda: "DEVICE = torch.device("cuda")"
MAX_MEMORY_SIZE = 10240
BATCH_SIZE = 2560

# THREAD_NUM base on EPSILON_LIST
# each tuple in EPSILON_LIST is (ROBOT_A_EPSILON, ROBOT_A_EPSILON_DECAY, ROBOT_B_EPSILON, ROBOT_B_EPSILON_DECAY)
EPSILON_LIST = [
    (0.8, 0.99, 0.3, 1),
    (0.8, 0.99, 0.4, 1),
    (0.8, 0.99, 0.5, 1),
    (0.8, 0.99, 0.6, 1),
    (0.8, 0.99, 0.7, 1),
    (0.8, 0.99, 0.8, 1),
    (0.8, 0.99, 0.5, 0.99),
    (0.8, 0.99, 0.5, 0.99),
    (0.8, 0.99, 0.7, 0.99),
    (0.8, 0.99, 0.7, 0.99),
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
    # board_size and lr are meaningless for the next two rebot.
    # set max_memory_size=0 is avoiding to train at these two robots.
    robot_A = agent.gobang.robot(
        device=DEVICE,
        epsilon=robot_a_episode,
        epsilon_decay=robot_a_episode_decay,
        board_size=BOARD_SIZE,
        lr=LEARNING_RATE,
        max_memory_size=0
    )
    robot_B = agent.gobang.robot(
        device=DEVICE,
        epsilon=robot_b_episode,
        epsilon_decay=robot_b_episode_decay,
        board_size=BOARD_SIZE,
        lr=LEARNING_RATE,
        max_memory_size=0
    )

    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)
    for epoch in range(TRAIN_TIME):
        start_next_game.acquire(blocking=True)

        # get module from tol_robot
        robot_A.change_module_from_other(tol_robot)
        robot_B.change_module_from_other(tol_robot)

        env.clear()
        cnt = 0
        while True:
            done = robot_step(env.A, robot_A, env, memorize_to_robot=tol_robot, is_train=False)
            cnt += 1
            if done != 0:
                who_win = done
                break

            done = robot_step(env.B, robot_B, env, memorize_to_robot=tol_robot, is_train=False)
            cnt += 1
            if done != 0:
                who_win = done
                break

        game_info.put((cnt, who_win))

        game_over_count.release(1)


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

    tol_robot.save(MODULE_SAVE_PATH)

    atexit.unregister(when_unexpect_exit)


if __name__ == '__main__':
    main()
