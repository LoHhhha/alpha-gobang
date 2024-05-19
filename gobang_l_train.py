import atexit
import os.path
from collections import deque

import numpy as np
import torch
import time
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader

import agent
import environment
from agent.module.L_Net_v2 import L_Net
from torchvision import transforms

from data.data import MyDataset
import matplotlib.pyplot as plt

TRAIN_TIME = 20000
BOARD_SIZE = 3
WIN_SIZE = 3
MODULE_SAVE_PATH = "./best_gobang.pth"
MODULE_UE_SAVE_PATH = "best_gobang_ue.pth"
CURRENT_MODULE_SAVE_PATH = "./params/current_gobang.pth"
PATH = "./params"
LEARNING_RATE = 0.0001
SHOW_BOARD_TIME = 100
DEVICE = torch.device("cuda")  # if you wait to use cuda: "DEVICE = torch.device("cuda")"
ROBOT_A_EPSILON = 0.8
ROBOT_A_EPSILON_DECAY = 0.99
ROBOT_B_EPSILON = 0.3
ROBOT_B_EPSILON_DECAY = 1
MAX_SIZE = 4096
TRAIN_DIS = 1
SAVE_NAME = '_3_3_3_V2_loss_dict.pth'
TRAIN_REAPET_TIME=1
PLAY_PATH = "./params/8500test1_dict.pth"
LOAD = False
LOAD_PATH = "./params/10000_3_3_3_V2_loss_dict.pth"
LEARNING_RATE_DEC = 0.99

loss_fun = torch.nn.CrossEntropyLoss()

def show_loss(loss):
    x = list(range(len(loss)))
    plt.plot(x, loss)

    # 添加标题和标签
    plt.title('Loss')
    plt.xlabel('times')
    plt.ylabel('loss')

    # 显示图表
    plt.show()
def robot_step(
        who, env, robot=None, modlue=None,
        memorize_to_robot=None,
        show_result: bool = False,
        board_size: int = BOARD_SIZE
):
    if (modlue == None):
        params = torch.load(PLAY_PATH)
        modlue = L_Net(board_size)
        modlue.load_state_dict(params)
        modlue = modlue.to(DEVICE)
        modlue.eval()
    state = env.get_state(who)
    if robot == None:
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
    else:
        action = robot.get_action(state)
        best_place = torch.argmax(action).item()
        r, c = best_place // board_size, best_place % board_size
        env.step(who, (r, c))
        done = env.check_game_end()
        return done
    if show_result:
        print(f"chosen:{best_place}:{action.detach().cpu().numpy()}")
    r, c = best_place // board_size, best_place % board_size
    env.step(who, (r, c))
    done = env.check_game_end()

    # 设置label
    action = action.cpu().detach().numpy()
    act_win = torch.tensor(action)
    act_win[0, best_place] = 1
    act_loss = torch.tensor(action)
    act_loss[0, best_place] = 0

    if memorize_to_robot == 'A':
        memorize_to_robotA_real_list.append(torch.tensor(action))
        memorize_to_robotA_state_list.append(state)
        memorize_to_robotA_loss_list.append(act_loss)
        memorize_to_robotA_win_list.append(act_win)

    return done

loss_list=[]
memorize_to_robotA_state_list = []
memorize_to_robotA_real_list = []
memorize_to_robotA_loss_list = []
memorize_to_robotA_win_list = []
state_list = deque(maxlen=MAX_SIZE)
act_list = deque(maxlen=MAX_SIZE)


def init_memorize():
    memorize_to_robotA_real_list.clear()
    memorize_to_robotA_state_list.clear()
    memorize_to_robotA_loss_list.clear()
    memorize_to_robotA_win_list.clear()


def clear_train_data():
    state_list.clear()
    act_list.clear()


def get_label_old(A_win, A_loss, A_real, who_win):
    l = len(A_win)
    rate = 1
    w = 1
    A_act = []
    if who_win == "A":
        for i in range(l):
            A_act.append(F.softmax((A_win[l - i - 1] - A_real[l - i - 1]) * w + A_real[l - i - 1]))
            w *= rate
    if who_win == "B":
        for i in range(l):
            A_act.append(F.softmax((A_loss[l - i - 1] - A_real[l - i - 1]) * w + A_real[l - i - 1]))
            w *= rate
    if who_win == "C":
        memorize_to_robotA_state_list.clear()
    A_act.reverse()
    return A_act

def get_label(A_win, A_loss, A_real, who_win):
    l = len(A_win)
    rate = 0.9
    w = 1
    A_act = []
    if who_win == "A":
        for i in range(l):
            sumnum=sum(((A_win[i ] - A_real[i ]) * w + A_real[ i ])).sum()
            A_act.append(((A_win[i ] - A_real[i ]) * w + A_real[i])/sumnum)
            w *= rate
    if who_win == "B":
        for i in range(l):
            sumnum=((A_loss[i ] - A_real[i ]) * w + A_real[ i ]).sum()
            add=(A_loss[i ] - A_real[i ]) * w + A_real[ i ]
            A_act.append(add/sumnum)
            w *= rate
    if who_win == "C":
        for i in range(l):
            sumnum=sum(((A_win[i ] - A_real[i ]) * w + A_real[ i ])).sum()
            A_act.append(((A_win[i ] - A_real[i ]) * w + A_real[i])/sumnum)
            w *= rate
    return A_act


def train_with_memory(net, device, state_list, act_list, lr):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if(len(state_list)==0):
        return
    data_loader=DataLoader(MyDataset(state_list,act_list,board_size=BOARD_SIZE), batch_size=64, shuffle=False)
    for (input_data, label) in data_loader:
        optimizer.zero_grad()
        input_data, label = input_data.to(device), label.to(device)
        output = net(input_data)
        train_loss = loss_fun(output, label)
        train_loss.backward()
        loss_list.append(train_loss.item())
        optimizer.step()


def train():
    lr = LEARNING_RATE
    if (not os.path.exists(PATH)):
        os.mkdir("params")
    l_module = L_Net(board_size=BOARD_SIZE)
    l_module = l_module.to(DEVICE)

    best_module = L_Net(board_size=BOARD_SIZE)
    best_module = best_module.to(DEVICE)
    if LOAD:
        params = torch.load(LOAD_PATH)
        l_module.load_state_dict(params)
        best_module.load_state_dict(params)

    @atexit.register
    def when_unexpect_exit():
        torch.save(l_module.state_dict(), CURRENT_MODULE_SAVE_PATH)

    env = environment.gobang.game(board_size=BOARD_SIZE, win_size=WIN_SIZE)

    dm_robotB = agent.gobang_dm.dm_robot(env.B, env, display_reward=False)
    random_robotB = agent.gobang_randon.randon_robot(env.B, env, display_reward=False)

    avg_time = 0
    for epoch in range(TRAIN_TIME):
        init_memorize()
        start_time = time.time()

        who_win = "C"
        env.clear()
        while True:

            done = robot_step(env.A, env, modlue=l_module, memorize_to_robot="A", )
            if done == 1:
                who_win = "C"
                best_module.load_state_dict(l_module.state_dict())
                break
            elif done != 0:
                who_win = "C"
                break

            done = robot_step(env.B, env, modlue=best_module, memorize_to_robot="B", )
            if done == 1:
                who_win = "B"
                break
            elif done != 0:
                who_win = "C"
                break
        if epoch % SHOW_BOARD_TIME == 0:
            env.display()
            print(who_win)
            print(done)
        act_list.extend(
            get_label(memorize_to_robotA_win_list, memorize_to_robotA_loss_list, memorize_to_robotA_real_list,
                      who_win))
        state_list.extend(memorize_to_robotA_state_list)

        init_memorize()
        env.clear()
        while True:
            done = robot_step(env.B, env, modlue=best_module, memorize_to_robot="B", )
            if done == 1:
                who_win = "B"
                break
            elif done != 0:
                who_win = "C"
                break

            done = robot_step(env.A, env, modlue=l_module, memorize_to_robot="A", )
            if done == 1:
                who_win = "C"
                best_module.load_state_dict(l_module.state_dict())
                break
            elif done != 0:
                who_win = "C"
                break
        act_list.extend(
            get_label(memorize_to_robotA_win_list, memorize_to_robotA_loss_list, memorize_to_robotA_real_list,
                      who_win))
        state_list.extend(memorize_to_robotA_state_list)
        if epoch % SHOW_BOARD_TIME == 0:
            env.display()
            print(who_win)
            print(done)

        init_memorize()
        env.clear()
        while True:
            done = robot_step(env.B, env, robot=dm_robotB, modlue=best_module, memorize_to_robot="B", )
            if done == 1:
                who_win = "B"
                break
            elif done != 0:
                who_win = "C"
                break

            done = robot_step(env.A, env, modlue=l_module, memorize_to_robot="A", )
            if done == 1:
                who_win = "C"
                break
            elif done != 0:
                who_win = "C"
                break
        act_list.extend(
            get_label(memorize_to_robotA_win_list, memorize_to_robotA_loss_list, memorize_to_robotA_real_list,
                      who_win))
        state_list.extend(memorize_to_robotA_state_list)
        if epoch % SHOW_BOARD_TIME == 0:
            env.display()
            print(who_win)
            print(done)

        # init_memorize()
        # env.clear()
        # while True:
        #     done = robot_step(env.A, env, modlue=l_module, memorize_to_robot="A", )
        #     if done == 1:
        #         who_win = "C"
        #         break
        #     elif done != 0:
        #         who_win = "C"
        #         break
        #     done = robot_step(env.B, env, robot=dm_robotB, modlue=best_module, memorize_to_robot="B", )
        #     if done == 1:
        #         who_win = "B"
        #         break
        #     elif done != 0:
        #         who_win = "C"
        #         break
        # act_list.extend(
        #     get_label(memorize_to_robotA_win_list, memorize_to_robotA_loss_list, memorize_to_robotA_real_list,
        #               who_win))
        # state_list.extend(memorize_to_robotA_state_list)
        # if epoch % SHOW_BOARD_TIME == 0:
        #     env.display()
        #     print(who_win)
        #     print(done)
        #
        # init_memorize()
        # env.clear()
        # while True:
        #     done = robot_step(env.A, env, modlue=l_module, memorize_to_robot="A", )
        #     if done == 1:
        #         who_win = "C"
        #         break
        #     elif done != 0:
        #         who_win = "C"
        #         break
        #     done = robot_step(env.B, env, robot=random_robotB, modlue=best_module, memorize_to_robot="B", )
        #     if done == 1:
        #         who_win = "B"
        #         break
        #     elif done != 0:
        #         who_win = "C"
        #         break
        # act_list.extend(
        #     get_label(memorize_to_robotA_win_list, memorize_to_robotA_loss_list, memorize_to_robotA_real_list,
        #               who_win))
        # state_list.extend(memorize_to_robotA_state_list)
        # if epoch % SHOW_BOARD_TIME == 0:
        #     env.display()
        #     print(who_win)
        #     print(done)

        init_memorize()
        env.clear()
        while True:
            done = robot_step(env.B, env, robot=random_robotB, modlue=best_module, memorize_to_robot="B", )
            if done == 1:
                who_win = "B"
                break
            elif done != 0:
                who_win = "C"
                break

            done = robot_step(env.A, env, modlue=l_module, memorize_to_robot="A", )
            if done == 1:
                who_win = "C"
                break
            elif done != 0:
                who_win = "C"
                break
        act_list.extend(
            get_label(memorize_to_robotA_win_list, memorize_to_robotA_loss_list, memorize_to_robotA_real_list,
                      who_win))
        state_list.extend(memorize_to_robotA_state_list)
        if epoch % SHOW_BOARD_TIME == 0:
            env.display()
            print(who_win)
            print(done)

        if epoch % TRAIN_DIS ==0:
            for i in range(TRAIN_REAPET_TIME):
                train_with_memory(l_module, DEVICE, state_list, act_list, lr)
                lr = lr * LEARNING_RATE_DEC
            clear_train_data()
        diff_time = time.time() - start_time
        avg_time = 0.5 * (avg_time + diff_time)
        print(f"Epoch {epoch + 1}/{TRAIN_TIME}, {diff_time:.3f}it/s, {avg_time * (TRAIN_TIME - epoch - 1):.0f}s left:")
        if epoch % 100 == 0:
            show_loss(loss_list)
            torch.save(l_module.state_dict(), PATH + "/" + str(epoch) + SAVE_NAME)
            torch.save(l_module.state_dict(), PATH + "/best_" + str(epoch) + SAVE_NAME)
    atexit.unregister(when_unexpect_exit)


if __name__ == '__main__':
    train()
