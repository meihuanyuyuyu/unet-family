import torch
from torch.nn import modules
import utils
from torch.nn.modules import module
from utils.augmentation import Random_crop,Random_flip,Randomrotation,My_center_crop,My_colorjitter,Normalize_,Elastic_deformation
from torch.utils.data import Dataset
from utils import train,val,save_result_json
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm
from models import *
from config import *
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

writer = SummaryWriter()

def main(i,
        train_t:list,
        val_t:list,
        model:nn.Module,
        e:int,
        dataset:Dataset):
    train_sampled,val_sampled = utils.split_train_test(dataset,batch_size,train_t,val_t)
    net =model(in_c=in_c,num_class=num_class,depth=depth,is_bn=bn,active_func=nn.LeakyReLU).to(device)
    net.apply(init_weight)
    opt = SGD(net.parameters(),lr=lr,momentum=0.9)
    criterion = utils.Dice_loss_with_logist()
    model_dir,json_dir =utils.auto_path(net,i)
    val_f1s = []
    losses = []
    f1s = []
    bar = tqdm(range(e))
    for epoch in bar:
        loss,f1 = train(net,train_sampled,criterion,opt,device=device)
        writer.add_scalar('Loss/train', loss, epoch)
        bar.set_description(f'training loss:{loss},f1 score:{f1}')
        val_f1 = val(i,net,val_sampled,device=device)
        bar.set_description(f'val_f1 {val_f1}')
        losses.append(loss)
        val_f1s.append(val_f1)
        f1s.append(f1)
        writer.add_scalars('variation of two metric', {'f1':f1,'val_f1':val_f1}, epoch)
        if epoch % 100 ==0:
            torch.save(net.state_dict(),model_dir)
    writer.close()
    save_result_json(json_dir,i,losses,val_f1s,f1s)
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(9,[Random_crop([512,512]),My_colorjitter(0.7,0.7,0.7),Elastic_deformation(10,100),Randomrotation([0.1,20.1]),Random_flip(),Normalize_([0.5,0.5,0.5],[0.5,0.5,0.5])],
    [My_center_crop([512,512]),Normalize_([0.5,0.5,0.5],[0.5,0.5,0.5])],unet,1000,utils.Mousegment_2018_dataset)
