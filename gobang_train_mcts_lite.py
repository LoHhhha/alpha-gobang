import torch

import agent.gobang
import environment
from gobang_train import robot_step

SIM_NUM = 100
SEARCH_NODE = 2
BOARD_SIZE = 3
WIN_SIZE = 3
NODE_VALUE_FROM_DM = True
SMALL_P_NODE_RANDOM_SELECT_RATE = 0
MAX_MEMORY_SIZE = 5120
BATCH_SIZE = 512
LEARNING_RATE = 0.0000005
GAMMA = 0.5
DEVICE = torch.device('cpu')
LOSS_FUNC_CLASS = torch.nn.MSELoss

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
    for i in range(SIM_NUM):
        simulate()
        print("OK")


if __name__ == '__main__':
    main()
