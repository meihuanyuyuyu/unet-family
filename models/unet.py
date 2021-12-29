from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn


class Down(nn.Module):
    r'下采样模块，输入(n,c,h,w)的特征图，依次输出下采样得到的特征图和用来跳跃连接特征图，in_channel:输入下采样模块的通道数，out_channel:输出模块的通道数'

    def __init__(self, in_channel: int, out_channel: int, is_bn: Optional[bool] = None, active_func=nn.ReLU) -> None:
        super().__init__()
        ################### 初始化一些卷积参数 ###################################
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.is_bn = is_bn
        if is_bn:
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel), active_func(True), nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), active_func(True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=3, padding=1), active_func(True), nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1), active_func(True))
        self.maxpool = nn.AvgPool2d(2)

    def forward(self, x):
        ################### 前向传播 ############################################
        x1 = x = self.conv1(x)
        return self.maxpool(x), x1  # 返回下采样后的张量x和跳跃连接张量x1


class UpSample(nn.Module):
    r'上采样模块，输入两个变量，一个是顺基础块正向传播的特征图和跳跃连接过来的特征图，依次输出下采样得到的特征图和用来跳跃连接特征图，in_c:输入下采样模块的通道数，out_c:输出模块的通道数'

    def __init__(self, in_c, out_c, is_bn: Optional[bool] = False, active_func=nn.ReLU):
        super().__init__()
        ################### 初始化一些卷积参数 ###################################
        self.in_c = in_c
        self.out_c = out_c
        if is_bn:
            self.conv2 = nn.Sequential(nn.Conv2d(self.in_c, self.out_c, 3, 1, 1), nn.BatchNorm2d(out_c), active_func(True), nn.Conv2d(self.out_c, self.out_c, 3, 1, 1), nn.BatchNorm2d(out_c), active_func(True))
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(self.in_c, self.out_c, 3, 1, 1), active_func(True), nn.Conv2d(self.out_c, self.out_c, 3, 1, 1), active_func(True))
        self.up = nn.ConvTranspose2d(self.out_c, int(self.out_c / 2), 2, 2)

    def forward(self, x, x1):
        x = torch.cat([x, x1], dim=1)  # 通道数进行拼接
        x = self.conv2(x)
        return self.up(x)


class mid_bridge(nn.Module):
    r'桥接层'
    def __init__(self, in_c, out_c, is_bn: Optional[bool] = False, active_func=nn.ReLU) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        if is_bn:
            self.conv = nn.Sequential(nn.Dropout(), nn.Conv2d(self.in_c, self.out_c, 3, 1, 1), nn.BatchNorm2d(out_c), active_func(), nn.Conv2d(self.out_c, self.out_c, 3, 1, 1), nn.BatchNorm2d(out_c), active_func())
        else:
            self.conv = nn.Sequential(nn.Dropout(), nn.Conv2d(self.in_c, self.out_c, 3, 1, 1), active_func(), nn.Conv2d(self.out_c, self.out_c, 3, 1, 1),  active_func())
        self.up = nn.ConvTranspose2d(self.out_c, int(self.out_c / 2), 2, 2)

    def forward(self, x):
        x = self.conv(x)
        return self.up(x)


class unet(nn.Module):
    def __init__(self,in_c,num_class,depth:int=5,is_bn:Optional[bool]=None,active_func=nn.ReLU):
        super().__init__()
        self.depth = depth
        filters = self.auto_channels(depth)
        assert(depth>1)
        for i in range(1,depth):
            down = Down(in_c,filters[i-1],is_bn,active_func)
            setattr(self,f'down{i}',down)
            in_c = filters[i-1]
        self.mid = mid_bridge(filters[-2], filters[-1],is_bn,active_func)
        for i in range(2,depth):
            up = UpSample(filters[i],filters[i-1],is_bn,active_func)
            setattr(self,f'up{i}',up)
        self.final = nn.Sequential(nn.Conv2d(filters[1], filters[0], 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, num_class, 1))
    
    
    def auto_channels(self,depth):
        n = 64
        filters =[]
        for i in range(depth):
            if i!=0:
                n = 2*n
            filters.append(n)
        return filters


    def forward(self, x: Tensor):
        skip_path = []
        for i in range(1,self.depth): # 下采样层数为depth-1
            down =getattr(self,f'down{i}')
            x,x_n = down(x)
            skip_path.append(x_n) # 生成的跳跃连结对应索引关系：index=0，skip_path[index] -> 第一层下采样层输出的结果  
        x = self.mid(x)
        for i in range(self.depth-1,1,-1): # 从第depth-1层到第二层，将下采样产生的跳跃连结与之前上采样输出，送进上采样层
            up = getattr(self,f'up{i}')
            x = up(x,skip_path[i-1])
        x = torch.cat([x, skip_path[0]], dim=1)
        return self.final(x)
