# -*- coding: UTF-8 -*- 
# Creator：LeK
# Date：2024/5/13
import pygame
from tkinter import messagebox
import sys
import torch
import agent
import environment
from gobang_train import robot_step

GRID_SIZE = 3  # 棋盘格子数
CELL_SIZE = 40  # 单元格大小
WIN_SIZE = 3  # 胜利数
HUMAN_COLOR = 2  # 执棋颜色 黑1 白2 黑子先手
BOARD_COLOR = (255, 206, 158)  # 棋盘底色
MODULE_PATH = None
DEMO_PATH = './demo.txt'
OPTION = 0  # 游玩 0 demo演示 1


class GobangGame:
    def __init__(self, grid_size=GRID_SIZE, cell_size=CELL_SIZE):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_length = grid_size * cell_size
        self.screen = pygame.display.set_mode((self.grid_length, self.grid_length))
        pygame.display.set_caption("Alpha Gobang Game")
        self.env = environment.gobang.game(board_size=GRID_SIZE, win_size=WIN_SIZE)
        self.player_human = self.env.A if HUMAN_COLOR == 1 else self.env.B
        self.player_robot = self.env.A if HUMAN_COLOR == 2 else self.env.B
        if MODULE_PATH is None:
            self.robot = agent.gobang_dm.dm_robot(self.player_robot, self.env, display_reward=False)
        else:
            self.robot = agent.gobang.robot(module_save_path=MODULE_PATH, epsilon=0, board_size=GRID_SIZE)
            self.robot.module.eval()
        self.is_pause = False

    # 绘制棋盘
    def draw_board(self):
        for i in range(self.grid_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * self.cell_size), (self.grid_length, i * self.cell_size))
            pygame.draw.line(self.screen, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, self.grid_length))

    # 绘制棋子
    def draw_piece(self, row, col):
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        if self.env.board[row][col] == 1:
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), self.cell_size // 2 - 2)
        elif self.env.board[row][col] == -1:
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), self.cell_size // 2 - 2)

    # 绘制游戏画面
    def display(self):
        self.screen.fill(BOARD_COLOR)
        self.draw_board()
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.env.board[row][col] != 0:
                    self.draw_piece(row, col)
        pygame.display.flip()

    # 落子
    def place_piece(self, row, col):
        with torch.no_grad():
            self.env.step(self.player_human, (row, col))
            self.display()

            done = self.env.check()

            # computer can place
            if done == 0:
                done = robot_step(self.player_robot, self.robot, self.env, is_train=False, show_result=True,
                                  board_size=GRID_SIZE)

            if done == 0:
                pass
            if done == self.player_robot:
                self.display()
                print("Computer wins!")
                messagebox.showinfo(title="游戏结束", message="Computer wins!")
                self.is_pause = True
            elif done == self.player_human:
                self.display()
                print("You win!")
                messagebox.showinfo(title="游戏结束", message="You win!")
                self.is_pause = True
            elif done == self.env.draw_play:
                self.display()
                print("Draw!")
                messagebox.showinfo(title="游戏结束", message="Draw!")
                self.is_pause = True

            self.display()

    def run(self):
        if OPTION == 0:
            if HUMAN_COLOR == 2:
                robot_step(self.env.A, self.robot, self.env, is_train=False, show_result=True, board_size=GRID_SIZE)
            self.display()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if self.is_pause:
                        continue
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        row, col = y // self.cell_size, x // self.cell_size
                        if 0 <= row < self.grid_size and 0 <= col < self.grid_size and self.env.board[row][col] == 0:
                            self.place_piece(row, col)
        else:
            # TODO:输入序列
            with open(DEMO_PATH, 'r') as f:
                for line in f:
                    pass


if __name__ == "__main__":
    pygame.init()
    game = GobangGame()
    game.run()
