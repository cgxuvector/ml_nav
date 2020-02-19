"""
    Customized transformation
"""
import torch
from skimage.transform import resize


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, sample):
        # covert from numpy: H x W x C to tensor: C x H x W
        # add one more dimension to local map
        if self.mode == 'cls':
            sample['localmap'] = torch.from_numpy(sample['localmap'])[None]
            for idx, s in enumerate(sample['observations']):
                sample['observations'][idx] = torch.from_numpy(s.transpose(2, 0, 1))
        elif self.mode == 'cls-iid':
            image = sample["observations"]
            label = sample["label"]

            image = resize(image, (64, 64, 3)).transpose(2, 0, 1)
            sample = {"observations": torch.from_numpy(image), "label": torch.tensor(label)}
        return sample
