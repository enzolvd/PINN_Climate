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

class ERADataset(Dataset):
    """Custom Dataset for ERA climate data."""
    
    def __init__(
        self,
        root_dir,
        nb_file, 
        years,
        vars = base_vars):

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

                    

