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
            for idx, s in enumerate(sample['observation']):
                sample['observation'][idx] = torch.from_numpy(s.transpose(2, 0, 1))
        elif self.mode == 'cls-iid':
            image = sample["observation"]
            label = sample["label"]

            image = resize(image, (64, 64, 3)).transpose(2, 0, 1)
            sample = {"observations": torch.from_numpy(image), "label": torch.tensor(label)}
        elif self.mode == 'cvae':
            obs_image = sample["observation"]
            map_image = sample["loc_map"]
            ori = sample["orientation"]
            # note: image is convert between [0, 1]
            obs_image = resize(obs_image, (64, 64, 3)).transpose(2, 0, 1)
            map_image = resize(map_image, (3, 3))
            sample = {"observation": torch.from_numpy(obs_image),
                      "loc_map": torch.tensor(map_image).unsqueeze(0),
                      "orientation": torch.from_numpy(ori).unsqueeze(0)}
            assert True
        return sample
