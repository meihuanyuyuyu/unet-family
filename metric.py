from torch import Tensor
import torch
def F1score_white_as_target(binarymap:Tensor,labels:Tensor)->None:
    TP = ((binarymap==1)&(labels==1)).sum().item()
    FP_FN = torch.ne(binarymap,labels).sum().item()
    return TP/(TP+FP_FN*0.5)

def F1score_black_as_target(binarymap:Tensor,labels:Tensor):
    TP = ((binarymap==0)&(labels==0)).sum().item()
    FP_FN = torch.ne(binarymap,labels).sum().item()
    return TP/(TP+FP_FN*0.5)

def jaccard_black_as_target(binarymap:Tensor,labels:Tensor)->None:
    l1 = labels +binarymap
    TP = (l1==0).sum().item()
    FP_FN = torch.ne(binarymap,labels).sum().item()
    return TP/(TP+FP_FN)

def jaccard_white_as_target(binarymap:Tensor,labels:Tensor)->None:
    l1 = labels +binarymap
    TP = (l1==2).sum().item()
    FP_FN = torch.ne(binarymap,labels).sum().item()
    return TP/(TP+FP_FN)

def precision(binarymap:Tensor,label:Tensor,tp:int=0)->float:
    erro_msg = 'tp arg must be 0 or 1'
    assert (tp == 1)or(tp == 0), erro_msg
    TP = ((binarymap==tp)&(label==tp)).sum().item()
    TP_FP = (binarymap == tp).sum().item()
    return TP/TP_FP

def recall(binarymap:Tensor,label:Tensor,tp:int=0):
    erro_msg = 'tp arg must be 0 or 1'
    assert (tp == 1)or(tp == 0), erro_msg
    TP = ((binarymap==tp)and(label==tp)).sum().item()
    TP_FN = (label==tp).sum().item()
    return TP/TP_FN

