import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import xarray as xr
from pathlib import Path
import glob
import logging
import os

base_vars = {
        'geopotential_500':'z',
        '10m_u_component_of_wind':'u10',
        '10m_v_component_of_wind':'v10',
        '2m_temperature':'t2m',
        'temperature_850':'t'
}

input_target_repartition = {
    'input' : ['geopotential_500', 'temperature_850'],
    'target' : ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
}

class MeteoDataset(Dataset): 

    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        """Return the total number of samples."""
        return len(self.x)
    
    def __getitem__(self, idx):
        """Return one sample of data with its label."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_input = self.input[idx]
        sample_target = self.target[idx]
                    
        return sample_input, sample_target

class ERADataset(Dataset):
    """Custom Dataset for ERA climate data."""
    
    def __init__(
        self,
        root_dir,
        nb_file, 
        years,
        vars = base_vars, 
        input_target_repartition = input_target_repartition, 
        train_val_split = None):

        super().__init__()

        self.root_dir = Path(root_dir)
        self.nb_file = nb_file
        self.years = years
        self.vars = vars

        # Load grid file
        try:
            grid_path = self.root_dir / 'constants' / 'constants.nc'
            self.grid_ds = xr.open_dataset(grid_path, engine='netcdf4')
        except Exception as e:
            logging.error(f"Error loading constants files: {e}")

        self.data_vars = {}

        # Load dataset
        for var in self.vars.keys():
            data = []
            list_file_var = os.listdir(self.root_dir / var )
            key = self.vars[var]
            for year in years:
                try:
                    file = [f for f in list_file_var if f'_{str(year)}_' in f][0]
                    data_path = self.root_dir / var / file
                    data.append(xr.open_dataset(data_path)[key].to_numpy())
                except Exception as e:
                    logging.error(f'No file found at year {year}: {e}')

            self.data_vars [var] = np.concatenate(data)

        if not train_val_split: # only load for train:
            input = []
            for key in input_target_repartition['input']:
                input.append(torch.from_numpy(self.data_vars[key]))
            input = torch.stack(input, axis=-1)

            target = []
            for key in input_target_repartition['target']:
                target.append(torch.from_numpy(self.data_vars[key]))
            target = torch.stack(target, axis=-1)
            self.dataset = Dataset(input, target)

                    

