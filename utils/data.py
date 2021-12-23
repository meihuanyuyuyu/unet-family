import os
from typing import Sequence
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
class Mydataset(Dataset):
    def __init__(self, root, mode='train', transforms: list = None) -> None:
        super().__init__()
        if mode == 'train':
            self.path = os.path.join(root, '2018 Training Data')
            self.images_path = os.path.join(self.path, 'Tissue Images')
            self.labels_path = os.path.join(self.path, 'Annotations')
        if mode == 'test':
            self.path = os.path.join(root, 'MoNuSegTestData')
            self.images_path = os.path.join(self.path, 'imgs')
            self.labels_path = os.path.join(self.path, 'labels')
        self.images = os.listdir(self.images_path)
        self.labels = os.listdir(self.labels_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def transform(self, img, label):
        if self.transforms:
            for i in self.transforms:
                img, label = i(img, label)
        return img, label

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        img = Image.open(os.path.join(self.images_path, img))
        img = F.to_tensor(img)
        label = Image.open(os.path.join(self.labels_path, label))
        label = F.to_tensor(label)
        return self.transform(img, label)

class split_set(Dataset):
    def __init__(self, data: Dataset, indices: Sequence[int], transforms: list = None) -> None:
        super().__init__()
        self.data = data
        self.indices = indices
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.data[self.indices[index]]
        return self.transform(img, label)

    def transform(self, img: Tensor, label: Tensor):
        if self.transforms:
            for i in self.transforms:
                img, label = i(img, label)
        return img, label

    def __len__(self):
        return len(self.indices)

def kfold_crossval(data: Dataset, k: int = 4, num_fold: int = 0, train_t=[], val_t=[]):
    '''
    返回两个划分好的训练和验证集，数据类型为Dataset类
    '''
    assert (num_fold < k)
    length = len(data)
    k_size = int(length / k)
    test_arr = [num_fold * k_size + i for i in range(k_size)]
    return split_set(data, [x for x in range(length) if x not in test_arr], train_t), split_set(data, test_arr, val_t)

class EM_mydataset(Dataset):
    def __init__(self, root: str, transforms: list=None) -> None:
        super().__init__()
        self.img_path = os.path.join(root, 'train/image')
        self.label_path = os.path.join(root, 'train/label')
        self.labels = os.listdir(self.label_path)
        self.imgs = os.listdir(self.img_path)
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir('./EMdata/train/image'))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index]
        label = Image.open(os.path.join(self.label_path, label))
        img = Image.open(os.path.join(self.img_path, img))
        img, label = F.to_tensor(img), F.to_tensor(label)
        return self.transform(img, label)

    def transform(self, img: Tensor, label: Tensor):
        if self.transforms:
            for i in self.transforms:
                img, label = i(img, label)
        return img, label

class Mousegment_2018_dataset(Dataset):
    def __init__(self,root:str,mode:str='train',transform:list=None) -> None:
        super().__init__()
        if mode == 'train':
            self.imgs_path = os.path.join(root,'2018 Training Data/Tissue Images')
            self.labels_path = os.path.join(root,'2018 Training Data/Annotations')
            self.imgs = os.listdir(self.imgs_path)
            self.labels = os.listdir(self.labels_path)
        if mode == 'val':
            self.imgs_path = os.path.join(root,'MoNuSegTestData/imgs')
            self.labels_path = os.path.join(root,'MoNuSegTestData/labels')
            self.imgs = os.listdir(self.imgs_path)
            self.labels = os.listdir(self.labels_path)
        self.transforms =transform
    
    def __len__(self):
        return len(self.imgs)

    def transform(self,img,label):
        if not self.transforms:
            return img,label
        else:
            for t in self.transforms:
                img,label = t(img,label)
            return img,label
    
    def __getitem__(self, index):
        img = os.path.join(self.imgs_path,self.imgs[index])
        label = os.path.join(self.labels_path,self.labels[index])
        img=Image.open(img)
        label = Image.open(label)
        img,label = F.to_tensor(img),F.to_tensor(label)
        return self.transform(img,label)

