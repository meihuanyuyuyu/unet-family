k = 5 # k折交叉验证，如果没启用交叉验证，此参数可不管
batch_size = 2 # 批量大小
lr = 9e-3 # 学习率大小
bn = True # 模型是否使用batchnormalize
in_c = 3 # 模型输入通道数，列如rgb图片为3
num_class = 2 # 输出类别
depth = 5 # 模型层数