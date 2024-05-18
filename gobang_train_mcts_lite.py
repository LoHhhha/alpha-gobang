import atexit
import datetime

import torch

import agent.gobang
import environment
from gobang_train import robot_step

SIM_NUM = 100
SEARCH_NODE = 3
BOARD_SIZE = 3
WIN_SIZE = 3
NODE_VALUE_FROM_DM = True
DRAW_PLAY_IS_WIN = False
SMALL_P_NODE_RANDOM_SELECT_RATE = 0.7
MAX_MEMORY_SIZE = 5120
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
GAMMA = 0.2
DEVICE = torch.device('cpu')
LOSS_FUNC_CLASS = torch.nn.SmoothL1Loss

STATR_TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
MODULE_SAVE_PATH = (f"./alpha_gobang_B{BOARD_SIZE}_W{WIN_SIZE}_"
                    f"{STATR_TIME}_mcts.pth")
MODULE_UE_SAVE_PATH = (f"./alpha_gobang_B{BOARD_SIZE}_W{WIN_SIZE}_"
                       f"{STATR_TIME}_mcts_ue.pth")

torch.manual_seed(19528)

env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)
robot_mc = agent.gobang_mc.mc_robot(
    board_size=BOARD_SIZE,
    max_memory_size=MAX_MEMORY_SIZE,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    search_node_number=SEARCH_NODE,
    small_random_select_rate=SMALL_P_NODE_RANDOM_SELECT_RATE,
    value_from_dm=NODE_VALUE_FROM_DM,
    draw_play_is_win=DRAW_PLAY_IS_WIN,
    gamma=GAMMA,
    device=DEVICE,
    loss_class=LOSS_FUNC_CLASS
)


def simulate():
    env.clear()
    robot_mc.module.train()
    while True:
        robot_mc.search_and_get_experience(env, env.A)
        robot_mc.train_memory()
        done = robot_step(env.A, robot_mc, env, memorize_to_robot=None, is_train=False, board_size=BOARD_SIZE,
                          show_result=True)
        env.display()
        if done != 0:
            who_win = done
            break

        robot_mc.search_and_get_experience(env, env.B)
        robot_mc.train_memory()
        done = robot_step(env.B, robot_mc, env, memorize_to_robot=None, is_train=False, board_size=BOARD_SIZE,
                          show_result=True)
        env.display()
        if done != 0:
            who_win = done
            break


def main():
    @atexit.register
    def when_unexpect_exit():
        torch.save(robot_mc.module, MODULE_UE_SAVE_PATH)
        print("[note] because unexpected exit, we save current net as '{}'.".format(MODULE_UE_SAVE_PATH))

    for i in range(SIM_NUM):
        simulate()
        print(f"OK {i + 1}/{SIM_NUM}")

    torch.save(robot_mc.module, MODULE_SAVE_PATH)
    atexit.unregister(when_unexpect_exit)


if __name__ == '__main__':
    main()
