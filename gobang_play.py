import torch

import agent
import environment
from gobang_train import robot_step

BOARD_SIZE = 3
WIN_SIZE = 3
MODULE_SAVE_PATH = "./best_gobang_multi_ue.pth"


def play(board_size: int, win_size: int, module_path: str):
    robot = agent.gobang.robot(device=torch.device('cpu'), epsilon=0, board_size=board_size,
                               module_save_path=module_path)

    env = environment.gobang.game(board_size=board_size, win_size=win_size)
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


def play_with_dm(board_size: int, win_size: int):
    env = environment.gobang.game(board_size=board_size, win_size=win_size)
    robot = agent.gobang_dm.dm_robot(env.A, env, display_reward=True)

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
    play(BOARD_SIZE, WIN_SIZE, MODULE_SAVE_PATH)
    play_with_dm(BOARD_SIZE, WIN_SIZE)
