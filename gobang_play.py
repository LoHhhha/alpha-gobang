import torch

import agent
import environment
from gobang_train import robot_step

BOARD_SIZE = 8
WIN_SIZE = 5
MODULE_SAVE_PATH = "./xxx.pth"


def play(board_size: int, win_size: int, module_path: str):
    robot = agent.gobang.robot(device=torch.device('cpu'), epsilon=0, board_size=board_size,
                               module_save_path=module_path)
    robot.module.eval()

    env = environment.gobang.game(board_size=board_size, win_size=win_size)
    with torch.no_grad():
        while True:
            done = robot_step(env.A, robot, env, is_train=False, show_result=True, board_size=BOARD_SIZE)

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
            done = robot_step(env.A, robot, env, is_train=False, show_result=True, board_size=BOARD_SIZE)  # 模型执黑下

            if done != 0:  # 游戏结束
                break

            env.display()

            while True:  # 输入要下的棋子位置
                a = int(input("r->"))
                b = int(input("c->"))
                if env.board[a][b] != 0:  # 输入不合法，重新输入
                    continue
                env.step(env.B, (a, b))  # 输入成功
                if env.pre_action is not None:
                    break

            if env.check() != 0:  # 检查游戏是否结束
                break

        env.display()


if __name__ == '__main__':
    # play(BOARD_SIZE, WIN_SIZE, MODULE_SAVE_PATH)
    play_with_dm(BOARD_SIZE, WIN_SIZE)
