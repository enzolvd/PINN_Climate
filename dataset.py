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

class ERADataset(Dataset):
    """Custom Dataset for ERA climate data with optional normalization."""
    
    def __init__(
        self,
        root_dir,
        years,
        normalize = True,
        vars = base_vars, 
        input_target_repartition = input_target_repartition,
        norm_params = None):  # Add norm_params parameter

        super().__init__()

        self.root_dir = Path(root_dir)
        self.years = years
        self.vars = vars
        self.normalize = normalize
        
        # Use provided norm_params or create new ones
        self.norm_params = norm_params if norm_params is not None else {}

        # Load grid file
        try:
            grid_path = self.root_dir / 'constants' / 'constants.nc'
            self.grid_ds = xr.open_dataset(grid_path, engine='netcdf4')
        except Exception as e:
            logging.error(f"Error loading constants files: {e}")

        # Extract coordinates and ensure float32 precision
        self.lons = self.grid_ds['lon2d'].to_numpy().astype(np.float32)
        self.lats = self.grid_ds['lat2d'].to_numpy().astype(np.float32)
        
        # Store coordinate normalization parameters only if not provided
        if not self.norm_params:
            self.norm_params['lon'] = {
                'min': float(self.lons.min()),
                'max': float(self.lons.max())
            }
            self.norm_params['lat'] = {
                'min': float(self.lats.min()),
                'max': float(self.lats.max())
            }
        
        if self.normalize:
            self.lons = ((self.lons - self.norm_params['lon']['min']) / 
                        (self.norm_params['lon']['max'] - self.norm_params['lon']['min'])).astype(np.float32)
            self.lats = ((self.lats - self.norm_params['lat']['min']) / 
                        (self.norm_params['lat']['max'] - self.norm_params['lat']['min'])).astype(np.float32)
        
        self.data_vars = {}        

        # Load dataset
        for var in self.vars.keys():
            self.data = []
            list_file_var = os.listdir(self.root_dir / var)
            key = self.vars[var]
            for year in years:
                try:
                    file = [f for f in list_file_var if f'_{str(year)}_' in f][0]
                    data_path = self.root_dir / var / file
                    self.data.append(xr.open_dataset(data_path)[key].to_numpy().astype(np.float32))
                except Exception as e:
                    logging.error(f'No file found at year {year}: {e}')

            self.data_vars[var] = np.concatenate(self.data)
            
            # Store normalization parameters only if not provided
            if not self.norm_params.get(var):
                self.norm_params[var] = {
                    'min': float(self.data_vars[var].min()),
                    'max': float(self.data_vars[var].max())
                }
            
            # Normalize if requested and ensure float32 precision
            if self.normalize:
                self.data_vars[var] = ((self.data_vars[var] - self.norm_params[var]['min']) / 
                                     (self.norm_params[var]['max'] - self.norm_params[var]['min'])).astype(np.float32)

        # Load constant masks
        self.constant_masks = {}
        for key in ['orography', 'lsm', 'slt']:
            data = self.grid_ds[key].to_numpy().astype(np.float32)
            if not self.norm_params.get(key):
                self.norm_params[key] = {
                    'min': float(data.min()),
                    'max': float(data.max())
                }
            if self.normalize:
                data = ((data - self.norm_params[key]['min']) / 
                       (self.norm_params[key]['max'] - self.norm_params[key]['min'])).astype(np.float32)
            self.constant_masks[key] = torch.from_numpy(data).float()
        self.constant_masks = torch.stack(list(self.constant_masks.values()))

        # Load and process time
        time = []
        for year in years:
            list_file_var = os.listdir(self.root_dir / 'geopotential_500')
            file = [f for f in list_file_var if f'_{str(year)}_' in f][0]
            data_path = self.root_dir / 'geopotential_500' / file
            time.append(xr.open_dataset(data_path)['time'].to_numpy())
            
        time = np.concatenate(time)
        self.time = ((time - np.datetime64('1979-01-01')) / np.timedelta64(1, 'h')).astype(np.float32)
        
        # Store time normalization parameters only if not provided
        if not self.norm_params.get('time'):
            self.norm_params['time'] = {
                'min': float(self.time.min()),
                'max': float(self.time.max())
            }
        
        if self.normalize:
            self.time = ((self.time - self.norm_params['time']['min']) / 
                        (self.norm_params['time']['max'] - self.norm_params['time']['min'])).astype(np.float32)
        
        # Prepare input and target tensors
        input_list = []
        for key in input_target_repartition['input']:
            input_list.append(torch.from_numpy(self.data_vars[key]).float())
        self.input = torch.stack(input_list, dim=-1).permute(0, 3, 1, 2)

        target_list = []
        for key in input_target_repartition['target']:
            target_list.append(torch.from_numpy(self.data_vars[key]).float())
        self.target = torch.stack(target_list, dim=-1).permute(0, 3, 1, 2)
        
        # Convert coordinates to torch tensors in float32
        self.lons = torch.from_numpy(self.lons).float()
        self.lats = torch.from_numpy(self.lats).float()
    
    def denormalize(self, data: torch.Tensor, var_name: str) -> torch.Tensor:
        """Denormalize data using stored parameters."""
        if not self.normalize:
            return data
        params = self.norm_params[var_name]
        return (data * (params['max'] - params['min']) + params['min']).float()  # Ensure float32
    
    def get_norm_params(self) -> Dict:
        """Return normalization parameters."""
        return self.norm_params
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return {
            'input': self.input[idx],  # [32, 64, 2]
            'target': self.target[idx],  # [32, 64, 3]
            'coords': [
                self.lons,  
                self.lats, 
                self.time[idx]],
            'masks': self.constant_masks,  # Dictionary of constant masks
            'norm_params': self.norm_params  # Normalization parameters
        }

def load_dataset(
    nb_file, 
    train_val_split = None, 
    year0=1979, 
    root_dir="./data/era_5_data",
    normalize=True):
    
    datasets = {
        'train': None,
        'val': None
    }
    
    if not train_val_split:
        years = np.arange(year0, year0 + nb_file, 1, dtype=np.int32)
        datasets['train'] = ERADataset(root_dir=root_dir, years=years, normalize=normalize)

    else:
        nb_train = np.floor(nb_file * train_val_split).astype(np.int32)
        nb_val = nb_file - nb_train
        years_train = np.arange(year0, year0 + nb_train, 1, dtype=np.int32)
        years_val = np.arange(year0 + nb_train, year0 + nb_train + nb_val, dtype=np.int32)
        
        # First create training dataset to get normalization parameters
        datasets['train'] = ERADataset(root_dir=root_dir, years=years_train, normalize=normalize)
        
        # Create validation dataset using training normalization parameters
        datasets['val'] = ERADataset(
            root_dir=root_dir, 
            years=years_val, 
            normalize=normalize,
            norm_params=datasets['train'].get_norm_params()  # Pass training normalization parameters
        )

    return datasets