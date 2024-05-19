# -*- coding: UTF-8 -*- 
# Creator：LeK
# Date：2024/5/13
import tkinter
import pygame
from tkinter import messagebox
from tkinter import filedialog
import sys
import torch
import agent
import environment
from gobang_train import robot_step
import datetime

GRID_SIZE = 3  # 棋盘格子数
WIN_SIZE = 3  # 胜利数
HUMAN_COLOR = 1  # 执棋颜色 黑1 白2 黑子先手
MODULE_PATH = None  # when MODULE_PATH is None, play with gobang_dm

OPTION = 0  # 游玩 0 demo演示 1

CELL_SIZE = 40  # 单元格大小
BOARD_COLOR = (255, 206, 158)  # 棋盘底色
IS_SAVE = True  # 是否记录步骤到回放文件中


class GobangGame:
    def __init__(self, grid_size=GRID_SIZE, cell_size=CELL_SIZE):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_length = grid_size * cell_size
        self.demo_path = None
        self.turn = 1
        if IS_SAVE:
            self.demo_save_path = (f'./alpha_gobang_demo_B{GRID_SIZE}_W{WIN_SIZE}_'
                                   f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt')
        self.env = environment.gobang.game(board_size=GRID_SIZE, win_size=WIN_SIZE)
        self.player_human = self.env.A if HUMAN_COLOR == 1 else self.env.B
        self.player_robot = self.env.A if HUMAN_COLOR == 2 else self.env.B
        if MODULE_PATH is None:
            self.robot = agent.gobang_dm.dm_robot(self.player_robot, self.env, display_reward=False)
        else:
            self.robot = agent.gobang.robot(module_save_path=MODULE_PATH, epsilon=0, board_size=GRID_SIZE)
            self.robot.module.eval()
        self.is_pause = False
        self.screen = pygame.display.set_mode((self.grid_length, self.grid_length))
        pygame.display.set_caption("Alpha Gobang Game")

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

    def save_step(self, row, col):
        if IS_SAVE:
            with open(self.demo_save_path, 'a') as f:
                f.write(str(row) + ',' + str(col) + '\n')

    def save_winner(self, winner):
        if IS_SAVE:
            with open(self.demo_save_path, 'a') as f:
                f.write(winner)

    def save_robot_step(self):
        state = self.env.get_state(self.player_robot)
        action = self.robot.get_action(state)
        place_hash = torch.argmax(action).item()
        self.save_step(place_hash // GRID_SIZE, place_hash % GRID_SIZE)

    # 落子
    def place_piece(self, row, col):
        with torch.no_grad():
            self.env.step(self.player_human, (row, col))
            self.display()
            self.save_step(row, col)
            done = self.env.check()

            # computer can place
            if done == 0:
                self.save_robot_step()
                done = robot_step(self.player_robot, self.robot, self.env, is_train=False, show_result=True,
                                  board_size=GRID_SIZE)
            if done == 0:
                pass
            if done == self.player_robot:
                self.display()
                print("Computer wins!")
                self.save_winner('Computer')
                messagebox.showinfo(title="游戏结束", message="Computer wins!")
                self.is_pause = True
            elif done == self.player_human:
                self.display()
                print("You win!")
                self.save_winner('You')
                messagebox.showinfo(title="游戏结束", message="You win!")
                self.is_pause = True
            elif done == self.env.draw_play:
                self.display()
                print("Draw!")
                self.save_winner('Nobody')
                messagebox.showinfo(title="游戏结束", message="Draw!")
                self.is_pause = True
            self.display()

    def run(self):
        if OPTION == 0:
            if HUMAN_COLOR == 2:
                self.save_robot_step()
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
            self.display()
            window = tkinter.Tk()
            window.withdraw()
            self.demo_path = filedialog.askopenfilename(title='选择回放文件')
            window.destroy()
            with open(self.demo_path, 'r') as f:
                idx = 0
                lines = [line.strip() for line in f]
                valid_lines = [line for line in lines if ',' in line]
                winner = next((line for line in lines if ',' not in line), None)
                row, col = zip(*(map(int, line.split(',')) for line in valid_lines))
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if self.is_pause:
                        continue
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.env.board[row[idx]][col[idx]] = 1 if self.turn == 1 else -1
                        self.turn = 1 if self.turn == 2 else 2
                        self.display()
                        idx += 1
                        if idx == len(row):
                            print(winner + " wins!")
                            messagebox.showinfo(title="游戏结束", message=winner + " wins!")
                            self.is_pause = True


if __name__ == "__main__":
    pygame.init()
    game = GobangGame()
    game.run()
