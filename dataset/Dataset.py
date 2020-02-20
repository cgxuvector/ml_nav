"""This file define the local map and observation dataset
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
import os
import fnmatch as fn
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
plt.rcParams.update({'font.size': 8})


class LocmapObsDataset(Dataset):
    """ Loal map and observation dataset"""
    # initialize the dataset
    def __init__(self, mode='cls', dir_path='/mnt/sda/dataset/ml_nav/loc_map_obs_fixed_texture', transform=None):
        self.mode = mode
        self.root_dir = dir_path
        self.transform = transform
        self.loc_map_name = fn.filter(os.listdir(dir_path), 'map_*')
        self.observation_frame_name = fn.filter(os.listdir(dir_path), 'obs_*')

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
        # self.orientation_angle = {"LOOK_NORTH_WEST": np.pi * 3 / 4,
        #                           "LOOK_NORTH": np.pi / 2,
        #                           "LOOK_NORTH_EAST": np.pi / 4,
        #                           "LOOK_WEST": np.pi,
        #                           "LOOK_EAST": 0.0,
        #                           "LOOK_SOUTH_WEST": np.pi * 5 / 4,
        #                           "LOOK_SOUTH": np.pi * 3 / 2,
        #                           "LOOK_SOUTH_EAST": np.pi * 7 / 4}

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
        if self.mode == "cls-iid" or self.mode == "cvae" or self.mode == "vae":
            return len(self.observation_frame_name)
        elif self.mode == "cls":
            return len(self.loc_map_name)
        else:
            assert False, "Error Mode: Please select the mode (cls-iid, cls, vae or cvae)"

    # get item
    def __getitem__(self, idx):
        # convert to python number
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # load the item based on mode
        if self.mode == "cls":
            loc_map = io.imread(self.root_dir + '/' + self.loc_map_name[idx])
            observations = []
            for ori in self.orientation_name:
                if ori == "GRAY.LOC_MAP":
                    continue
                # obs name
                obs_name = self.loc_map_name[idx].split('.')[0].replace('map_', 'obs_') + '_' + ori + '.png'
                observations.append(io.imread(self.root_dir + '/' + obs_name))
            # create a sample
            sample = {'observation': observations, 'localmap': loc_map}
        elif self.mode == "cls-iid":
            # file name
            obs_name = self.observation_frame_name[idx]
            loc_map_name = obs_name.replace("obs_", "map_").replace("_RGB."+obs_name.split('.')[1]+'.png', '.png')
            # data
            obs_img = io.imread(self.root_dir + '/' + obs_name)
            loc_map = io.imread(self.root_dir + '/' + loc_map_name).flatten() / 255
            # construct sample
            sample = {'observation': obs_img,
                      'label': loc_map[self.orientation_name.index('RGB.' + obs_name.split('.')[1])]}
        elif self.mode == "cvae":
            # file name
            obs_name = self.observation_frame_name[idx]
            loc_map_name = obs_name.replace("obs_", "map_").replace("_RGB." + obs_name.split('.')[1] + '.png', '.png')
            # data
            obs_img = io.imread(self.root_dir + '/' + obs_name)
            loc_map = io.imread(self.root_dir + '/' + loc_map_name)
            obs_ori = self.orientation_angle[obs_name.split(".")[1]]
            # construct sample
            sample = {'observation': obs_img,
                      'loc_map': loc_map,
                      'orientation': obs_ori}
        else:
            assert False, "Error Mode: Please select the mode (cls-iid or cls)"

        # apply transformation
        if self.transform:
            sample = self.transform(sample)

        return sample

    # display
    def show(self, sample):
        if self.mode == "cls":
            fig, arrs = plt.subplots(3, 3, figsize=(12, 12))
            fig.canvas.set_window_title("Observations")
            count = 0
            for i in range(3):
                for j in range(3):
                    if self.orientation_name[i * 3 + j] == 'GRAY.LOC_MAP':
                        arrs[i, j].set_title("Local Map")
                        if torch.is_tensor(sample['localmap']):
                            arrs[i, j].imshow(sample['localmap'].squeeze(0).squeeze(0).numpy())
                        else:
                            arrs[i, j].imshow(sample['localmap'])
                    else:
                        arrs[i, j].set_title(self.orientation_name[i * 3 + j])
                        if torch.is_tensor(sample['observations'][count]):
                            arrs[i, j].imshow(sample['observations'][count].squeeze(0).numpy().transpose(1, 2, 0))
                        else:
                            arrs[i, j].imshow(sample['observations'][count])
                        count += 1
            return fig
        elif self.mode == "cls-iid":
            image_num = sample["label"].size(0)
            fig, arr = plt.subplots(int(image_num / 2), 2, figsize=(6, 12))
            for idx in range(image_num):
                row = int(idx / 2)
                col = int(np.mod(idx, 2))
                arr[row, col].set_title("label : " + str(sample["label"][idx].item()))
                arr[row, col].imshow(sample["observations"][idx].squeeze(0).numpy().transpose(1, 2, 0))
            return fig
        else:
            assert False, "Error Mode: Please select the mode (cls-iid or cls)"


# transformed_dataset = LocmapObsDataset(mode="cls-iid", transform=transforms.Compose([Transform.ToTensor("cls-iid")]))
# dataLoader = DataLoader(transformed_dataset, batch_size=8, shuffle=True)
#
# for idx, batch in enumerate(dataLoader):
#     print(idx, ' - ', batch['observations'].size(), batch['label'].size())
#     fig = transformed_dataset.show(batch)
#     plt.show()

