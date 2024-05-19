# alpha-gobang

## Requirement

```
python>=3.12.2
torch>=2.2.2
pygame=2.5.2
```

tips: 更低版本的未作尝试，不代表不能运行。

## How to teach computer playing gobang? 怎么让计算机学会下五子棋？

### Reinforcement Learning 强化学习

* `agent` \ `environment` \ `action` \ `reward` \ `state`
* `agent`在特定的`environment`（以`state`量化）作出`action`可以得到对应的`reward`

## Solution 解决方案

### Deep Q-learning

```
python gobang_train.py
```

tips: 模型输出为棋子放于各点后的可能得到的最大分数

#### 关键超参数

| key                   | interpretation  |
|-----------------------|-----------------|
| TRAIN_TIME            | 训练轮次            |
| BOARD_SIZE            | 棋盘大小            |
| WIN_SIZE              | 顺利需要连成的棋子       |
| LEARNING_RATE         | 学习率             |
| DEVICE                | 训练所用的torch设备    |
| SHOW_BOARD_TIME       | 每训练多少轮后显示棋盘     |
| ROBOT_A_EPSILON       | robotA随机选择率     |
| ROBOT_A_EPSILON_DECAY | robotA随机选择率的衰减率 |
| ROBOT_B_EPSILON       | robotB随机选择率     |
| ROBOT_B_EPSILON_DECAY | robotB随机选择率的衰减率 |

#### 关键细节

* 使用`类Q-learning`策略
    * 对于当前某一位置的预测值，其应该是当前位置的`reward`+`未来可能得到的最大分数`
        * 本问题特殊在：
            * `agent`在一次`action`后需要另一个`agent`来进行一次`action`，所以`未来可能得到的最大分数`在本问题中被认为是*
              *对方能获得最大分数的相反数**（博弈论）而这个分数由自己的模型来估计。
            * `reward`难设置,目前将其设置为`environment.gobang.game.get_reward`
                * 占空格得分:对一个位置上的对角线、竖线、横线上的连续空格统计
                * 连续放置得分：对一个位置上的对角线、竖线、横线上的连续同色统计
                * 防守得分:对一个位置上的对角线、竖线、横线上的连续它色统计
                * 顺利得分:比其余多得多
                * [未加入]重复放置扣分
* 怎么组织训练?
    * 目前是分为`A`与`B`两个`agent`
    * `A`利用自身行为与`B`的行为完成拟合
    * `B`在`A`获胜时替换为`A`的参数
    * `A`预期是在前期多随机学习更多的行为,后期主要靠自身参数引导
    * `B`预期是在训练的全阶段都存在一定概率的随机,使得训练中持续存在更多的行为
    * 最终的训练状态是不断根据现有参数调优，完成对解的拟合。

### Deep Q-learning with Multithread

```
python gobang_train_multithread.py
```

#### 关键超参数

| key                | interpretation         |
|--------------------|------------------------|
| TRAIN_TIME         | 训练轮次                   |
| BOARD_SIZE         | 棋盘大小                   |
| WIN_SIZE           | 顺利需要连成的棋子              |
| LEARNING_RATE      | 学习率                    |
| DEVICE             | 训练所用的torch设备           |
| MAX_MEMORY_SIZE    | 最大的模型存储器大小             |
| BATCH_SIZE         | 训练的轮次                  |
| VALID_EPOCH        | 每训练多少轮后验证              |
| VALID_GAME_NUMBERS | 验证时进行的轮次数目             |
| EPSILON_LIST       | 对局的安排列表，此列表的大小等于训练线程数目 |

#### 关键细节

* 基本优化策略与`Deep Q-learning`一致
* 改进的点
    * 引入多线程，同时学习多对局，理论无上限
    * 引入决策树模型参与对局
    * 引入决策树评价模型`agent.gobang_dm.dm_robot`
* 对于各对局的设置
    * 最优模型对战全随机——增加局面学习
    * 最优模型对战最优模型
    * 最优模型对战决策树模型
    * 决策树模型对战决策树模型

1. 多战局同时启动
2. 得到所有战局中的行为
3. 训练整体模型
4. 回到`1`

### only win and loss

  ```
  python gobang_l_train.py
  ```

* 机器人A读入当前状态（State）输出一个ACTION
* 环境执行A操作
* 机器人B读入当前状态（State）输出一个ACTION
* 环境执行B操作
* 重复上面指导游戏结束
* 记录State和模型输出的Action
* 设置好胜利的Label以及失败的Label
    * 胜利置为奖励
    * 失败置为惩罚
* 根据游戏胜负选择对应的label进行训练

### Monte Carlo Search Tree Lite

  ```
  python gobang_train_mcts_lite.py
  ```

tips: 模型输出为棋子放于各点后的胜率

#### 关键超参数

| key                             | interpretation     |
|---------------------------------|--------------------|
| SIM_NUM                         | 训练轮次               |
| BOARD_SIZE                      | 棋盘大小               |
| WIN_SIZE                        | 顺利需要连成的棋子          |
| LEARNING_RATE                   | 学习率                |
| DEVICE                          | 训练所用的torch设备       |
| MAX_MEMORY_SIZE                 | 最大的模型存储器大小         |
| BATCH_SIZE                      | 训练的轮次              |
| SEARCH_NODE                     | 每层搜索节点数            |
| NODE_VALUE_FROM_DM              | 节点的权重是否来自于决策树给定的价值 |
| DRAW_PLAY_IS_WIN                | 平局时是否计入胜利          |
| SMALL_P_NODE_RANDOM_SELECT_RATE | 随机选择小权重点的比例        |
| GAMMA                           | 新计算出来的胜率更新到期望胜率的权重 |
| LOSS_FUNC_CLASS                 | 损失函数的选择            |

#### 关键细节

* 简化版的蒙特卡洛算法
* 训练阶段
    * 对局面上的若干个空位置搜索至游戏结束
    * 自底向上完成状态胜率的估计
    * 优化模型输出
* 推理阶段
    * 不再搜索
    * 选择模型最大胜率点下棋子

## How to evaluate the model we got? 怎么评价得到的模型？

* GUI
  ```
  python gobang_play_gui.py
  ```

* CLI
  ```
  python gobang_play.py
  ```