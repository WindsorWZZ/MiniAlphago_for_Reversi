import json
import os
from glob import glob
from logging import getLogger
from args import ResourceConfig

import torch

logger = getLogger(__name__)
from torch.utils.data.dataset import Dataset

def get_play_data_filenames(rc: ResourceConfig):
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files

def get_next_generation_model_dirs(rc: ResourceConfig):
    dir_pattern = os.listdir(rc.next_generation_model_dir)
    dirs = [os.path.join(rc.next_generation_model_dir, i) for i in sorted(dir_pattern)]
    return dirs

def write_game_data_to_file(path, data):
    if data:
        with open(path, "wt") as f:
            json.dump(data, f)

def read_game_data_from_file(path):
    with open(path, "rt") as f:
        return json.load(f)


class PlayDataset(Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self,  state_ary, policy_ary, z_ary):
        # init
        self.state = state_ary
        self.policy = policy_ary
        self.z = z_ary

    def __len__(self):
        # len
        return (self.state.shape[0])

    def __getitem__(self, idx):
        # get item
        return self.state[idx], self.policy[idx], self.z[idx]