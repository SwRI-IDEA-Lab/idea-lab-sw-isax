import torch
from torch.utils.data import Dataset
import datetime as dt
import cdflib

class SolarWindDataset(Dataset):
    def __init__(self,
                 fname,
                 parameter:str = 'flow_speed',):
        cdf_file = cdflib.CDF(fname)
        self.parameter = parameter
        self.data = cdf_file[parameter]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]