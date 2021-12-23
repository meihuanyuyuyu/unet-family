from genericpath import exists
from re import I
from .data import Mydataset,EM_mydataset,Mousegment_2018_dataset,split_set,kfold_crossval
from .loss import Dice_loss_with_logist,Dice_loss
from .augmentation import Random_crop,Random_flip,Resize,Elastic_deformation,Normalize_,CenterCrop,ColorJitter
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import save_image
import numpy as np
from torch import Tensor
import json
import torch
import metric

def train(model: nn.Module, sampled_data: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str = 'cuda'):
    losses = []
    f1s = []
    model.train()
    for data in sampled_data:
        imgs, labels = data
        labels = labels.squeeze(1)
        outputs = model(imgs.to(device))
        optimizer.zero_grad()
        # white = labels.sum()/(labels.size(0)*labels.size(1)*labels.size(2))
        #white = white.item()
        #black = 1-white
        #torch.Tensor([white,black]).to(device)
        # loss = criterion(outputs,labels.to(device=device,dtype=torch.long),torch.Tensor([white,black]).to(device))
        loss = criterion(outputs, labels.to(device, dtype=torch.long))
        binarymap = torch.argmax(outputs, dim=1)
        loss.backward()
        optimizer.step()
        f1 = metric.F1score_white_as_target(binarymap, labels.to(device))
        f1s.append(f1)
        losses.append(loss.item())
        return np.array(losses).mean(), np.array(f1s).mean()

def val(i, model: nn.Module, sampled_data: DataLoader, device: str = 'cuda'):
    f1s = []
    model.eval()
    for data in sampled_data:
        imgs, labels = data
        labels = labels.squeeze(1)
        outputs = model(imgs.to(device))
        binarymap = torch.argmax(outputs, dim=1)
        f1score = metric.F1score_white_as_target(binarymap, labels.to(device))
        f1s.append(f1score)
        visualization_white_as_target(binarymap, labels, imgs, f'{i}val.png')
    return np.array(f1s).mean()

def visualization_black_as_target(predict: Tensor, label: Tensor, name) -> None:
    size = predict.size()
    result = torch.zeros(size[0], 3, size[1], size[2])
    result[:, 0, :, :] = 1 - label
    result[:, 1, :, :] = 1 - predict
    save_image(result, os.path.join('vision_result', name))

def visualization_white_as_target(predict: Tensor, label: Tensor, input: Tensor, name):
    size = predict.size()
    result = torch.zeros(size[0], 3, size[1], size[2])
    result[:, 0, :, :] = label
    result[:, 1, :, :] = predict
    save_fp = os.path.join(os.getcwd(),'vision_result')
    if not os.path.exists(save_fp):
        os.makedirs(save_fp)
    save_image(input, os.path.join(save_fp, 'img_' + name))
    save_image(result, os.path.join(save_fp, name))

def save_result_json(fp: str, i, loss, val_f1, f1s):
    if not os.path.exists(fp):
        os.makedirs(fp)
    result = {'i': i, 'loss': loss, 'f1': f1s, 'val_f1': val_f1}
    with open(os.path.join(fp, f'_{i}_train_score.json'), 'w') as f:
        json.dump(result, f)

def auto_path(model:nn.Module,i):
    result_root = os.path.join(os.getcwd(),'results')
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    if model.__class__.__name__== 'attention_u_net':
        model_para_path = os.path.join(result_root,'model_parameters/attention unet')
        json_path = os.path.join(result_root,'json/attention unet')
        if not os.path.exists(json_path):
            os.makedirs(json_path)
        if not os.path.exists(model_para_path):
            os.makedirs(model_para_path)
        model_dir = os.path.join(result_root,f'model_parameters/attention unet/{i}_EM_model.pt')
        json_dir = os.path.join(result_root,f'training_json/attention unet')
    if model.__class__.__name__ =='unet':
        model_para_path = os.path.join(result_root,'model_parameters/unet')
        json_path = os.path.join(result_root,'json/unet')
        if not os.path.exists(json_path):
            os.makedirs(json_path)
        if not os.path.exists(model_para_path):
            os.makedirs(model_para_path)
        model_dir = os.path.join(result_root,f'model_parameters/unet/{i}_EM_model.pt')
        json_dir = os.path.join(result_root,f'training_json/unet')
    if model.__class__.__name__ =='residual_unet':
        model_para_path = os.path.join(result_root,'model_parameters/residual unet')
        json_path = os.path.join(result_root,'json/residual unet')
        if not os.path.exists(json_path):
            os.makedirs(json_path)
        if not os.path.exists(model_para_path):
            os.makedirs(model_para_path)
        model_dir = os.path.join(result_root,f'model_parameters/residual unet/{i}_EM_model.pt')
        json_dir = os.path.join(result_root,f'training_json/residual unet')
    return model_dir,json_dir

def split_train_test(dataset:Dataset,batch_size,train_t,val_t,i:int=0,kfold__mode:bool=False,k=5):
    root_path = os.path.join(os.getcwd(),'MICCAI2018MoNuSeg')
    if kfold__mode:
        data = dataset(root_path,[])
        t,v= kfold_crossval(data,k,i,train_t,val_t)
        return DataLoader(t,batch_size=batch_size,shuffle=True,num_workers=1),DataLoader(v,batch_size=batch_size,shuffle=True,num_workers=1)
    else:
        train_set = dataset(root_path,'train',train_t)
        val_set = dataset(root_path,'val',val_t)
        return DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=1),DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=1)