# unet-family: Ultimate version
基于之前my-unet代码，我整理出来了这一份终极版本unet-family，方便其他人阅读。
1. 相比于之前的my-unet代码，代码分类更加规范，有条理
2. 对于clone下来的代码不需要修改各种复杂繁琐的路径问题，直接就可以运行。
3. 并且代码有很好的扩展性，可以增加各种模型，数据增强。
4. 接口设计易于修改各种参数，比如模型深度，激活函数，修改数据集类等，修改参数即可，代码自动适应网络架构。

模型只放上了u-net模型的模型架构。
## 1. 配置环境

在requirement.txt中导入所需要的工具包，可以pip install requirement.txt

## 2. 代码划分

代码分为五个部分,main,utils,mode,config,metric

### main.py
实现整个模型的基本逻辑
main参数:
i:设置k折交叉验证验证集选第几折，不使用k折交叉验证时表示第几次实验，方便记录
后两个参数是数据增强

model：是选用什么模型，设为unet

e： 训练轮数

dataset：选用什么数据类


### utils.py
包含数据增强，数据类，训练和验证基本逻辑

### model.py

里面写了unet模型的实现![](uTools_1640141465676.png)

### config.py
一些训练的参数，如批量大小，几折交叉验证k,还有数据集的路径。
