import torch
from torch import nn
from torch.nn import functional as F

# 输入一个（1，board_size*board_size)的tensor，输出也是（1，board_size*board_size)的tensor
# 但是要记得输入board_size，不然会报错
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, padding=0),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.03),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)

class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_channel, out_channel):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_channel, out_channel // 4, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_channel, in_channel // 4, kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channel // 4, out_channel // 4, kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接两个3 x 3卷积层
        self.p3_1 = nn.Conv2d(in_channel, in_channel // 4, kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=1)
        self.p3_3 = nn.Conv2d(in_channel // 4, out_channel // 4, kernel_size=3, padding=1)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channel, out_channel // 4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_3(self.p3_2(F.relu(self.p3_1(x)))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出


class L_Net(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.conv_size = [1, 8, 32, 32,128,128]
        self.conv1=Conv_Block(self.conv_size[0],self.conv_size[1])
        self.conv2=Inception(self.conv_size[1],self.conv_size[2])
        self.conv3=Conv_Block(self.conv_size[2],self.conv_size[3])
        self.conv4=Inception(self.conv_size[3],self.conv_size[4])
        self.conv5=Conv_Block(self.conv_size[4],self.conv_size[5])
        self.conv6=Conv_Block(self.conv_size[5],self.conv_size[4])
        self.conv7=Conv_Block(self.conv_size[4],self.conv_size[3])
        self.conv8=Conv_Block(self.conv_size[3],self.conv_size[2])
        self.conv9=Conv_Block(self.conv_size[2],self.conv_size[1])
        self.flaten=nn.Flatten()
        self.l=nn.Linear(self.conv_size[1]*self.board_size*self.board_size,self.board_size*self.board_size)
        self.sm=nn.Softmax()



    def forward(self, state):
        state=state.reshape(1,1,self.board_size,self.board_size)
        c1=self.conv1(state)
        c2=self.conv2(c1)
        c3=self.conv3(c2)
        c4=self.conv4(c3)
        c5=self.conv5(c4)
        c6=self.conv6(c5)
        c7=self.conv7(c6)
        c8=self.conv8(c7)
        c9=self.conv9(c8)
        c10=self.flaten(c9)
        res=self.l(c10)
        res=self.sm(res)
        return res


if __name__ == '__main__':
    _input = torch.randn(1, 9)
    net = L_Net(board_size=3)
    output = net(_input)
    print(output.shape)
