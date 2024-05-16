import atexit
import os.path

import numpy as np
import torch
import time
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader

import agent
import environment
from agent.module.L_Net import L_Net
from torchvision import transforms

TRAIN_TIME = 20000
BOARD_SIZE = 8
WIN_SIZE = 5
MODULE_SAVE_PATH = "./best_gobang.pth"
MODULE_UE_SAVE_PATH = "best_gobang_ue.pth"
CURRENT_MODULE_SAVE_PATH = "./params/current_gobang.pth"
PATH = "./params/"
LEARNING_RATE = 0.0001
SHOW_BOARD_TIME = 1
DEVICE = torch.device("cpu")  # if you wait to use cuda: "DEVICE = torch.device("cuda")"

ROBOT_A_EPSILON = 0.8
ROBOT_A_EPSILON_DECAY = 0.99
ROBOT_B_EPSILON = 0.3
ROBOT_B_EPSILON_DECAY = 1

loss_fun = torch.nn.CrossEntropyLoss()


def robot_step(
        who, env, modlue,
        memorize_to_robot=None,
        show_result: bool = False,
        board_size: int = BOARD_SIZE):
    state = env.get_state(who)
    state = torch.tensor(state, dtype=torch.float32)
    state = state.to(DEVICE)
    state = state.reshape(1, -1)
    action = modlue(state)
    action = action.unsqueeze(0)[0].detach()

    best_place = -1
    best_score = float('-inf')
    for i in range(state.size(1)):
        if state[0, i] == 0:
            if best_score < action[0, i]:
                best_score = action[0, i]
                best_place = i
        else:
            action[0, i] = 0

    if show_result:
        print(f"chosen:{best_place}:{action.detach().cpu().numpy()}")
    r, c = best_place // board_size, best_place % board_size
    env.step(who, (r, c))
    done = env.check_game_end()

    # 设置label
    action = action.cpu().detach().numpy()
    act_win = torch.zeros(action.shape)
    act_win[0, best_place] = 1
    act_loss = torch.tensor(action)
    act_loss[0, best_place] = -10

    if memorize_to_robot == 'A':
        memorize_to_robotA_state_list.append(state)
        memorize_to_robotA_loss_list.append(act_loss)
        memorize_to_robotA_win_list.append(act_win)
    if memorize_to_robot == 'B':
        memorize_to_robotB_state_list.append(state)
        memorize_to_robotB_loss_list.append(act_loss)
        memorize_to_robotB_win_list.append(act_win)
    return done


memorize_to_robotA_state_list = []
memorize_to_robotA_loss_list = []
memorize_to_robotA_win_list = []
memorize_to_robotB_state_list = []
memorize_to_robotB_loss_list = []
memorize_to_robotB_win_list = []


def init_memorize():
    memorize_to_robotA_state_list.clear()
    memorize_to_robotA_loss_list.clear()
    memorize_to_robotA_win_list.clear()
    memorize_to_robotB_state_list.clear()
    memorize_to_robotB_loss_list.clear()
    memorize_to_robotB_win_list.clear()


def train_with_memory(A_state, A_win, A_loss, B_state, B_win, B_loss, who_win, net, device):
    optimizer = optim.Adam(net.parameters())
    A_act = []
    B_act = []
    if who_win == "A":
        A_act = A_win
        B_act = B_loss
    if who_win == "B":
        A_act = A_loss
        B_act = B_win
    if who_win == "C":
        A_act = A_win
        B_act = B_win
    train_data = A_state + B_state
    target = A_act + B_act
    for i in range(len(train_data)):
        input_data, label = train_data[i], target[i]
        input_data, label = input_data.to(device), label.to(device)
        output = net(input_data)
        train_loss = loss_fun(output, label)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


def train():
    if(not os.path.exists("./params")):
        os.mkdir("params")
    l_module = L_Net(board_size=BOARD_SIZE*BOARD_SIZE)
    l_module = l_module.to(DEVICE)

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
        torch.save(l_module.state_dict(), CURRENT_MODULE_SAVE_PATH)

    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)

    avg_time = 0
    for epoch in range(TRAIN_TIME):
        init_memorize()
        start_time = time.time()
        env.clear()
        who_win = "C"
        while True:
            # if epoch % SHOW_BOARD_TIME == 0:
            #     print("A:", l_module(
            #         torch.tensor(env.get_state(env.A)).float().unsqueeze(0).to(DEVICE)).cpu().detach().numpy())

            # player1(train)
            done = robot_step(env.A, env, modlue=l_module, memorize_to_robot="A", )

            if done == 1:
                who_win = "A"
                break
            elif done!=0:
                break

            # player2(train)
            # if epoch % SHOW_BOARD_TIME == 0:
                # print("B:", l_module(
                #     torch.tensor(env.get_state(env.B)).float().unsqueeze(0).to(DEVICE)).cpu().detach().numpy())

            done = robot_step(env.B, env, modlue=l_module, memorize_to_robot="B", )
            if done == 1:
                who_win = "B"
                break
            elif done!=0:
                break
        train_with_memory(memorize_to_robotA_state_list,
                          memorize_to_robotA_win_list,
                          memorize_to_robotA_loss_list,
                          memorize_to_robotB_state_list,
                          memorize_to_robotB_win_list,
                          memorize_to_robotB_loss_list,
                          who_win,
                          l_module,
                          DEVICE)

        diff_time = time.time() - start_time
        avg_time = 0.5 * (avg_time + diff_time)
        print(f"Epoch {epoch + 1}/{TRAIN_TIME}, {diff_time:.3f}it/s, {avg_time * (TRAIN_TIME - epoch - 1):.0f}s left:")
        if epoch % SHOW_BOARD_TIME == 0:
            env.display()
            print(who_win)
            print(done)

        # if who_win == env.draw_play:
        #     print(f"draw after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")
        #     continue
        # if who_win == env.A:
        #     print(f"Player1 win after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")
        # if who_win == env.B:
        #     print(f"Player2 win after {BOARD_SIZE * BOARD_SIZE - env.count} step.\n")
        if(epoch%500==0):
            torch.save(l_module, PATH + str(epoch) + "_dict.pth")
    atexit.unregister(when_unexpect_exit)


if __name__ == '__main__':
    train()
