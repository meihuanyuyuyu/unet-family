import random
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
from torch.nn.functional import grid_sample
from PIL import Image
from torchvision.transforms.transforms import CenterCrop, ColorJitter

class Random_crop():
    '''
    随机裁剪。
    初始化输入裁剪大小.eg,[512,512]:list,
    调用类函数：传入变换的图片.img,label,...,图像类型可以为PIL或Tensor
    返回：裁剪完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor
    
    '''
    def __init__(self,size:list) -> None:
        super().__init__()
        self.size = size
        self.width = size[1]
        self.height = size[0] 
    
    def __call__(self,*args):
        param = T.RandomCrop.get_params(args[0],self.size)
        res = []
        for i in args:
            res.append(TF.crop(i,*param))
        return res

class Random_flip():
    '''
    随机翻转，0.5概率水平翻转，0.5概率垂直翻转
    初始化输入裁剪大小.eg,[512,512]:list,
    调用类函数：传入变换的图片.img,label,...list,图像类型可以为PIL或Tensor
    返回：翻转完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor

    '''
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self,*args):
        randomseed1 = random.random()
        res = []
        for i in args:
            if randomseed1<0.5:
                i = TF.vflip(i)
            if randomseed1<0.5:
                i= TF.hflip(i)
            res.append(i)
        return res

class Resize():
    '''
    放缩
    初始化输入大小.eg,[512,512]:list,
    调用类函数：传入变换的图片.img,label,...,图像类型可以为PIL或Tensor
    返回：放缩完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor   
    
    '''
    def __init__(self,size:list) -> None:
        super().__init__()
        self.size = size
    
    def __call__(self,*args:list):
        res = []
        for i in args:
            res.append(TF.resize(i,self.size))
        return res

class Randomrotation():
    '''
    随机旋转
    初始化输入翻转大小.eg,[0,359]，
    调用类函数：传入变换的图片.img,label,...,图像类型可以为PIL或Tensor
    返回：翻转完的图像列表.[img,label,...]:list,图像类型为PIL或Tensor   
    
    '''    
    def __init__(self,range) -> None:
        super().__init__()
        self.range = range
    
    def __call__(self,*args):
        res = []
        angel = T.RandomRotation.get_params(self.range)
        for i in args:
            res.append(TF.rotate(i,angel))
        return res

class Elastic_deformation():
    def __init__(self,sigma:int=10,grid_size:int =100,label_index:int=1,theta:float=0.5) -> None:
        super().__init__()
        self.sigma = sigma
        self.grid_size = grid_size
        self.label_index = label_index
        self.theta = theta

    def __call__(self,*args)->list[Tensor]:
        size = args[0].size()
        shift = torch.randn((2,int((size[-2]-1)/self.grid_size)+1,int((size[-1]-1)/self.grid_size)+1))*self.sigma
        shift = TF.resize(shift,(size[-2],size[-1])).permute(1,2,0)
        x = torch.linspace(-1,1,size[-1])
        y = torch.linspace(-1,1,size[-2])
        y,x = torch.meshgrid(y,x)
        y,x = y.unsqueeze(2),x.unsqueeze(2)
        xy = torch.cat((x,y),dim=2)
        shift[...,0] = shift[...,0]/size[-1]
        shift[...,1] = shift[...,0]/size[-2]
        xy = xy +shift
        res =[]
        for index,i in enumerate(args):
            if isinstance(i,Image.Image):
                i = TF.to_tensor(i)
            if index ==self.label_index:
                i =grid_sample(i.unsqueeze(0),xy.unsqueeze(0),'bicubic')
                i = (i>self.theta)*1.0
                res.append(i.squeeze(0))
            else:
                i =grid_sample(i.unsqueeze(0),xy.unsqueeze(0),'bicubic')
                res.append(i.squeeze(0))
        return res

class Normalize_():
    def __init__(self,mean:list,divation:list) -> None:
        super().__init__()
        self.mean = mean
        self.divation = divation
    def __call__(self,*args):
        assert isinstance(args[0],Tensor)
        res =[]
        for index,i in enumerate(args):
            if index ==0:
                res.append(TF.normalize(i,self.mean,self.divation))
            else:
                res.append(i)
        return res

class My_center_crop(CenterCrop):
    def __init__(self, size):
        super().__init__(size)
    
    def forward(self, *args):
        return [super().forward(_) for _ in args]

class My_colorjitter(ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self,*args):
        res = []
        for index,i in enumerate(args):
            if index==0:
                res.append(super().forward(args[0]))
            else:
                res.append(i)
        return res