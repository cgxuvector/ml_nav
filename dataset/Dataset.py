"""This file defines the dataset that contains the images of local maps and observations in 8 directions
    In customizing the dataset:
        library: from torch.utils.data import Dataset, DataLoader
                 from torchvision import transforms # transform.compose enable us to combine the customized transforms
        Customized classes:
                customized class has to inherit the Dataset and rewrite the two functions: def __len__ and def __getitem__
                customized transform has to define the callback function: def __call__(self, sample)
                in this dataset, we only need two transforms:
                    - Transform.ToTensor()
                    - Transform.Normalization()
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import os
import numpy as np
import fnmatch as fn
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
plt.rcParams.update({'font.size': 12})


class LocmapObsDataset(Dataset):
    """ The dataset
            Input:
                - mode: group, iid, conditional-iid
                - dir_path: path of the folder that contains the images
                - transform: data transform
            Output:
                - Customized dataset
    """
    # initialize the dataset
    def __init__(self,
                 mode='iid',
                 dir_path='/mnt/sda/dataset/ml_nav/global_map_obs_fixed_texture_small',
                 transform=None):
        self.mode = mode
        self.root_dir = dir_path
        self.transform = transform
        self.loc_map_name = fn.filter(os.listdir(dir_path), 'map_*')  # list of names of local maps images
        self.observation_frame_name = fn.filter(os.listdir(dir_path), 'obs_*')  # list of names of observation images

        # default orientation names
        self.orientation_name = ["RGB.LOOK_NORTH_WEST",
                                 "RGB.LOOK_NORTH",
                                 "RGB.LOOK_NORTH_EAST",
                                 "RGB.LOOK_WEST",
                                 "GRAY.LOC_MAP",
                                 "RGB.LOOK_EAST",
                                 "RGB.LOOK_SOUTH_WEST",
                                 "RGB.LOOK_SOUTH",
                                 "RGB.LOOK_SOUTH_EAST"]

        # orientation in scalars encoding
        self.orientation_angle = {"LOOK_NORTH_WEST": np.pi * 3 / 4,
                                  "LOOK_NORTH": np.pi / 2,
                                  "LOOK_NORTH_EAST": np.pi / 4,
                                  "LOOK_WEST": np.pi,
                                  "LOOK_EAST": 0.0,
                                  "LOOK_SOUTH_WEST": np.pi * 5 / 4,
                                  "LOOK_SOUTH": np.pi * 3 / 2,
                                  "LOOK_SOUTH_EAST": np.pi * 7 / 4}

        # orientations in one-hot encoding
        self.orientation_angle = {"LOOK_NORTH_WEST": np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                                  "LOOK_NORTH": np.array([0, 1, 0, 0, 0, 0, 0, 0]),
                                  "LOOK_NORTH_EAST": np.array([0, 0, 1, 0, 0, 0, 0, 0]),
                                  "LOOK_WEST": np.array([0, 0, 0, 1, 0, 0, 0, 0]),
                                  "LOOK_EAST": np.array([0, 0, 0, 0, 1, 0, 0, 0]),
                                  "LOOK_SOUTH_WEST": np.array([0, 0, 0, 0, 0, 1, 0, 0]),
                                  "LOOK_SOUTH": np.array([0, 0, 0, 0, 0, 0, 1, 0]),
                                  "LOOK_SOUTH_EAST": np.array([0, 0, 0, 0, 0, 0, 0, 1])}

    # obtain the length of the dataset
    def __len__(self):
        if self.mode == "iid" or self.mode == "conditional-iid":
            return len(self.observation_frame_name)
        elif self.mode == "group":
            return len(self.loc_map_name)
        else:
            assert False, "Error Mode: Please use the valid mode names from (group, iid, or conditional-iid)"

    # get item
    def __getitem__(self, idx):
        # convert to python number
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # group loading
        if self.mode == "group":
            # load the image of the local map
            loc_map = io.imread(self.root_dir + '/' + self.loc_map_name[idx])
            # load observations
            observations = []
            for ori in self.orientation_name:
                if ori == "GRAY.LOC_MAP":
                    continue
                # obs name
                obs_name = self.loc_map_name[idx].split('.')[0].replace('map_', 'obs_') + '_' + ori + '.png'
                observations.append(io.imread(self.root_dir + '/' + obs_name))
            # create a sample
            sample = {'observation': observations, 'loc_map': loc_map}
        # iid loading
        elif self.mode == "iid":
            # observation name
            obs_name = self.observation_frame_name[idx]
            # local map name
            loc_map_name = obs_name.replace("obs_", "map_").replace("_RGB."+obs_name.split('.')[1]+'.png', '.png')
            # load data of local map and observation
            obs_img = io.imread(self.root_dir + '/' + obs_name)
            loc_map = io.imread(self.root_dir + '/' + loc_map_name).flatten() / 255
            # construct sample
            sample = {'observation': obs_img,
                      'label': loc_map[self.orientation_name.index('RGB.' + obs_name.split('.')[1])]}
        elif self.mode == "conditional-iid":
            # name of the observation image
            obs_name = self.observation_frame_name[idx]
            # name of the local map image
            loc_map_name = obs_name.replace("obs_", "map_").replace("_RGB." + obs_name.split('.')[1] + '.png', '.png')
            # load data
            obs_img = io.imread(self.root_dir + '/' + obs_name)
            loc_map = io.imread(self.root_dir + '/' + loc_map_name)
            obs_ori = self.orientation_angle[obs_name.split(".")[1]]
            # construct sample
            sample = {'observation': obs_img,
                      'loc_map': loc_map,
                      'orientation': obs_ori}
        else:
            assert False, "Error Mode: Please use the valid mode names from (group, iid, or conditional-iid)"

        # apply transformation
        if self.transform:
            sample = self.transform(sample)

        return sample

    def split(self, set_size, trn_ratio, val_ratio, seed, shuffle=False):
        """
        Split the dataset into training, validation and testing
        :param set_size: size of the dataset
        :param trn_ratio: ratio of training samples
        :param val_ratio: ratio of validation samples
        :param seed: random seed
        :param shuffle: shuffle flag
        :return: samplers that can be used in dataLoader
        """
        indices = list(range(set_size))
        # compute the number of samples for training, validation, and testing
        trn_num = int(np.round(set_size * trn_ratio))
        val_num = int(np.round(set_size * val_ratio))
        tst_num = set_size - trn_num - val_num
        # shuffle the indices
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        # obtain the indices for training, validation, and testing
        trn_indices = indices[:trn_num]
        val_indices = indices[trn_num:trn_num + val_num]
        tst_indices = indices[set_size - tst_num:]

        trn_sample = SubsetRandomSampler(trn_indices)
        val_sample = SubsetRandomSampler(val_indices)
        tst_sample = SubsetRandomSampler(tst_indices)
        return trn_sample, val_sample, tst_sample

    # display
    def visualize_batch(self, sample, mode):
        if mode == "group":
            fig, arrs = plt.subplots(3, 3, figsize=(12, 12))
            fig.canvas.set_window_title("Panoramic Observations")
            count = 0
            for i in range(3):
                for j in range(3):
                    if self.orientation_name[i * 3 + j] == 'GRAY.LOC_MAP':
                        arrs[i, j].set_title("Local Map")  # set title for the local map window
                        # set data for numpy or tensor
                        if torch.is_tensor(sample['loc_map']):
                            arrs[i, j].imshow(sample['loc_map'].squeeze(0).squeeze(0).numpy())
                        else:
                            arrs[i, j].imshow(sample['loc_map'])
                    else:
                        # set title and data for the observations
                        arrs[i, j].set_title(self.orientation_name[i * 3 + j])
                        if torch.is_tensor(sample['observation'][count]):
                            arrs[i, j].imshow(sample['observation'][count].squeeze(0).numpy().transpose(1, 2, 0))
                        else:
                            arrs[i, j].imshow(sample['observation'][count])
                        count += 1
            return fig
        elif mode == "iid":
            image_num = sample["label"].size(0)  # obtain number of images in a mini-batch
            assert image_num >= 4, "Number of image should be even number and bigger than 4. " \
                                   "However, current is {}".format(image_num)
            if image_num > 1:
                fig, arr = plt.subplots(int(image_num / 2), 2, figsize=(6, 12))
            for idx in range(image_num):
                row = int(idx / 2)
                col = int(np.mod(idx, 2))
                arr[row, col].set_title("label : " + str(sample["label"][idx].item()))
                arr[row, col].imshow(sample["observation"][idx].squeeze(0).numpy().transpose(1, 2, 0))
            return fig
        elif mode == "conditional-iid":
            assert sample["orientation"].size(0) == 1, "Batch size should be 1"
            # get the orientation
            orientation = sample["orientation"]
            if torch.is_tensor(orientation):
                orientation = orientation.squeeze(0).numpy()[0].tolist()
            name_idx = orientation.index(1.0)
            orien_name = self.orientation_name[name_idx] if name_idx < 4 else self.orientation_name[name_idx + 1]
            # set up figure
            fig, arrs = plt.subplots(3, 3, figsize=(12, 12))
            fig.canvas.set_window_title("Conditional Panoramic Observations")
            for i in range(3):
                for j in range(3):
                    if self.orientation_name[i * 3 + j] == 'GRAY.LOC_MAP':
                        arrs[i, j].set_title("Map")
                        # set data for numpy or tensor
                        if torch.is_tensor(sample['loc_map']):
                            arrs[i, j].imshow(sample['loc_map'].squeeze(0).squeeze(0).numpy())
                        else:
                            arrs[i, j].imshow(sample['loc_map'])
                    elif self.orientation_name[i * 3 + j] == orien_name:
                        arrs[i, j].set_title(orien_name)
                        if torch.is_tensor(sample['observation']):
                            arrs[i, j].imshow(sample['observation'].squeeze(0).numpy().transpose(1, 2, 0))
                        else:
                            arrs[i, j].imshow(sample['observation'])
                    else:
                        arrs[i, j].imshow(np.zeros((64, 64, 3)))
            return fig
        else:
            assert False, "Error Mode: Please use the valid mode names from (group, iid, or conditional-iid)"


# """
#     Script test
# """
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
# from dataset import Transform
#
# # # mode name test
# # transformed_dataset = LocmapObsDataset(mode="sjdfajjd", transform=transforms.Compose([Transform.ToTensor("sadfds")]))
# # dataLoader = DataLoader(transformed_dataset, batch_size=8, shuffle=True)
#
# # loading test
# transformed_dataset = LocmapObsDataset(mode="group", transform=transforms.Compose([Transform.ToTensor("group")]))
# dataLoader = DataLoader(transformed_dataset, batch_size=1, shuffle=True)
#
# # test for first 50 elements
# for idx, batch in enumerate(dataLoader):
#     # x = batch["observation"]
#     # y = batch["loc_map"]
#     # z = batch["orientation"]
#     # print("Idx = ", idx+1, type(x), x.dtype, y.dtype, x.size(), y.size(), z.dtype, z.size())
#     fig = transformed_dataset.visualize_batch(batch, "group")
#     plt.show()
#     if idx == 9:
#         break

