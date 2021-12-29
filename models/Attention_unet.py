from typing import Optional
import torch.nn as nn
import torch
from .unet import Down,UpSample,mid_bridge


class attention_block(nn.Module):
    def __init__(self,gate_c,path_c,inter_c,is_bn:Optional[bool]=False,active_func=nn.ReLU) -> None:
        super().__init__()
        self.w_g = nn.Conv2d(gate_c,inter_c,kernel_size=1,stride=1)
        self.w_path = nn.Conv2d(path_c,inter_c,kernel_size=1,stride=1)
        if is_bn:
            self.att = nn.Sequential(
                nn.BatchNorm2d(inter_c),
                active_func(True),
                nn.Conv2d(inter_c,1,kernel_size=1,stride=1),
                nn.Sigmoid()
            )
        elif not is_bn:
            self.att = nn.Sequential(
                nn.BatchNorm2d(inter_c),
                active_func(True),
                nn.Conv2d(inter_c,1,kernel_size=1,stride=1),
                nn.Sigmoid()
            )
    
    def forward(self,x,path):
        x = self.w_g(x)
        path_x = self.w_path(path)
        a = self.att(x+path_x)
        return a*path
    

class attention_unet(nn.Module):
    def __init__(self,in_c,num_class,depth,is_bn:Optional[bool]=False,active_func=nn.ReLU,first_channel=64) -> None:
        super().__init__()
        self.depth = depth
        filters = [first_channel*2**_ for _ in range(depth)]
        for _ in range(1,depth):
            down = Down(in_c,filters[_-1],is_bn)
            setattr(self,f'down{_}',down)
            in_c = filters[_-1]
        self.mid = mid_bridge(filters[-2],filters[-1])
        for _ in range(2,depth):
            attention_gate = attention_block(filters[_-1],filters[_-1],filters[_-1],is_bn )
            up = UpSample(filters[_],filters[_-1])
            setattr(self,f'ag{_}',attention_gate)
            setattr(self,f'up{_}',up)
        self.final = nn.Sequential(
            nn.Conv2d(filters[1],filters[0],kernel_size=3,padding=1),
            nn.BatchNorm2d(filters[0]),
            active_func(True),
            nn.Conv2d(filters[0],filters[0],kernel_size=3,padding=1),
            nn.BatchNorm2d(filters[0]),
            active_func(True),
            nn.Conv2d(filters[0],num_class,kernel_size=1)
        )
    
    def forward(self,x):
        x_path = []
        for _ in range(1,self.depth):
            down=getattr(self,f'down{_}')
            x,x_g = down(x)
            x_path.append(x_g)
        x = self.mid(x)
        for _ in range(self.depth-1,1,-1):
            up = getattr(self,f'up{_}')
            ag = getattr(self,f'ag{_}')
            x_path[_-1] = ag(x,x_path[_-1])
            x = up(x,x_path[_-1])
        x =  torch.cat([x,x_path[0]],dim=1)
        return self.final(x)
            
            


        
 
        