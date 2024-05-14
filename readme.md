# alpha-gobang

## requirement

```
python>=3.12.2
torch>=2.2.2
pygame=2.5.2
```
注：更低版本的未作尝试，不代表不能运行。

## 细节

### 怎么开始

```
python gobang_train.py
```

### 训练方法

* 使用强化学习策略
  * `agent` \ `environment` \ `action` \ `reward` \ `state`
  * `agent`在特定的`environment`（以`state`量化）作出`action`可以得到对应的`reward`
* 使用`类Q-learning`策略
  * 对于当前某一位置的预测值，其应该是当前位置的`reward`+`未来可能得到的最大分数`
    * 本问题特殊在：
      * `agent`在一次`action`后需要另一个`agent`来进行一次`action`，所以`未来可能得到的最大分数`在本问题中被认为是**对方能获得最大分数的相反数**（博弈论）而这个分数由自己的模型来估计。
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

### 多线程版本`gobang_train_multithread`

1. 多战局同时启动
2. 得到所有战局中的行为
3. 训练整体模型
4. 回到`1`

#### 注意

* 新建线程的数目由`len(EPSILON_LIST)`给出
* `EPSILON_LIST`包含若干个元组，元组内元素为：`(ROBOT_A_EPSILON, ROBOT_A_EPSILON_DECAY, ROBOT_B_EPSILON, ROBOT_B_EPSILON_DECAY)`
* `EPSILON_LIST`应该尽可能多的考虑情况，给出不一样的随机策略