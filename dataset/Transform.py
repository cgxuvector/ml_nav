"""
    Customized transformation
"""
import torch
from skimage.transform import resize


class ToTensor(object):
    """
        Convert the data to tensor
    """
    def __init__(self, mode):
        # indicate different sample loader
        # "group": sample = {"observation": a list of observations in 8 directions, "loc_map": 3x3 local map}
        # "iid": sample = {"observation": an observation, "label": a label of the image}
        # "conditional-iid": sample = {"observation": an observation, "loc_map": 3x3 local map, "orientation": 1x8 one
        # hot encoding.}
        self.mode = mode

    def __call__(self, sample):
        # Functions:
        #   - covert image data from numpy: H x W x C to tensor: C x H x W
        #   - add one more dimension to local map
        #   - convert pixel value from [0, 255] int to [0, 1] float
        #   - convert label value from float to int
        if self.mode == 'group':
            sample['loc_map'] = torch.from_numpy(sample['loc_map']/255)[None].float()
            for idx, s in enumerate(sample['observation']):
                sample['observation'][idx] = torch.from_numpy(s.transpose(2, 0, 1) / 255).float()
        elif self.mode == 'iid':
            image = sample["observation"]
            label = sample["label"]
            # note: resize with rescale the pixel value to [0, 1]
            image = resize(image, (64, 64, 3)).transpose(2, 0, 1)
            sample = {"observation": torch.from_numpy(image).float(), "label": torch.tensor(label).long()}
        elif self.mode == 'conditional-iid':
            obs_image = sample["observation"]
            map_image = sample["loc_map"]
            ori = sample["orientation"]
            # note: image is convert between [0, 1]
            obs_image = resize(obs_image, (64, 64, 3)).transpose(2, 0, 1)
            map_image = resize(map_image, (3, 3))  # local map
            # map_image = resize(map_image, (21, 21))  # global map
            sample = {"observation": torch.from_numpy(obs_image).float(),
                      "loc_map": torch.tensor(map_image).unsqueeze(0).float(),
                      "orientation": torch.from_numpy(ori).unsqueeze(0).float()}
        else:
            assert False, "Mode Error: Please input valid mode. ('group', 'iid', or 'conditional-iid')"
        return sample
