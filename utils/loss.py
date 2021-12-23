import torch
from torch.functional import Tensor
from PIL import Image
import torch.nn.functional as nnf
import numpy as np
from scipy.ndimage import label, distance_transform_edt


class weighted_map(object):
    '''
    权重图
    '''
    def __init__(self, sigma=5, w_0=10) -> None:
        super().__init__()
        self.sigma = sigma
        self.w_0 = w_0

    def __call__(self, label_path: str) -> Tensor:
        l = np.array(Image.open(label_path), dtype=np.int64)
        print(l.shape)
        l = torch.tensor(l, dtype=torch.int64)
        l = l.clip(0, 1)
        cells, num_cells = label(l)
        cells = torch.tensor(cells, dtype=torch.int64)
        cells = nnf.one_hot(cells, num_classes=num_cells + 1)
        dists = torch.zeros_like(cells[..., 1:])
        for i in range(1, num_cells + 1):
            dist = cells[..., i].numpy()
            dists[..., i - 1] = torch.tensor(distance_transform_edt(1 - dist))
        dists, _ = dists.sort(dim=-1)
        d1 = dists[..., 0]
        d2 = dists[..., 1]
        w_c = l.sum().item() / (l.size(-2) * l.size(-1))
        return w_c + self.w_0 * torch.exp(-(d1 + d2)**2 / (2 * self.sigma**2)) * (l == 0)


class Dice_loss():
    def __init__(self, sigma: torch.long = 1e-8, reduction: str = 'mean') -> None:
        self.reduction = reduction
        self.sigma = sigma

    def _dice_loss(self, softmax: Tensor, target: Tensor):
        target = nnf.one_hot(target, num_classes=2)
        target = target.permute(0, 3, 1, 2)
        if self.reduction == 'mean':  # batch_size总的dice系数
            dice = 2 * (softmax * target).sum() / (softmax[:, 1].square().sum() + target.square().sum()) + self.sigma
            dice = dice
            return 1 - dice

    def __call__(self, predict: Tensor, target: Tensor):
        numerator = predict.exp()
        denominator = numerator.sum(dim=1, keepdim=True)
        softmax = numerator / denominator
        return self._dice_loss(softmax, target)


class Dice_loss_with_logist():
    def __init__(self) -> None:
        pass

    def _dice_loss_with_logist(self, predict: Tensor, target: Tensor):
        ce = nnf.cross_entropy(predict, target)
        target = nnf.one_hot(target, num_classes=2)
        target = target.permute(0, 3, 1, 2)
        target = target[:, 1]
        softmax = nnf.softmax(predict, dim=1)
        dice = 2 * (softmax[:, 1] * target).sum() / (softmax[:, 1].sum() + target.sum())
        dice = 1 - dice
        return ce / 2 + dice

    def __call__(self, predcit: Tensor, target: Tensor):
        return self._dice_loss_with_logist(predcit, target)
