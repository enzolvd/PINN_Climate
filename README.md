# Climate-PINN: Physics-Informed Neural Networks for Climate Modeling

This repository implements a Physics-Informed Neural Network (PINN) approach to climate modeling, combining deep learning with physical constraints from fluid dynamics. The model is trained on ERA5 climate data and predicts temperature and wind patterns while respecting fundamental physics principles.

## Features

- Deep learning model incorporating Navier-Stokes equations as physics constraints
- Training on ERA5 climate data with multiple variables (temperature, wind components, geopotential)
- Visualization tools for model predictions including animated temperature and wind field maps
- Automatic checkpointing and experiment tracking with Weights & Biases
- Support for distributed training on GPU clusters using SLURM

## Model Architecture

The Climate-PINN consists of several key components:

- **MeteoEncoder**: Processes meteorological input variables
- **MaskEncoder**: Handles geographical masks (orography, land-sea mask, etc.)
- **CoordProcessor**: Processes spatial and temporal coordinates
- **Physics Constraints**: Incorporates Navier-Stokes equations with learnable Reynolds number

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/climate-pinn.git
cd climate-pinn
```

2. Create and activate a conda environment:
```bash
conda create -n climate_pinn python=3.12
conda activate climate_pinn
```

3. Install required packages:
```bash
pip install torch xarray netCDF4 cartopy matplotlib wandb tqdm numpy
```

## Data Preparation

The model expects ERA5 climate data in NetCDF format with the following variables:
- Geopotential at 500 hPa
- Temperature at 2m and 850 hPa
- U and V wind components at 10m
- Constant fields (orography, land-sea mask, soil type)

Data should be organized in the following structure:
```
data/
└── era_5_data/
    ├── constants/
    │   └── constants.nc
    ├── geopotential_500/
    ├── 2m_temperature/
    ├── temperature_850/
    ├── 10m_u_component_of_wind/
    └── 10m_v_component_of_wind/
```

## Training

To train the model:

```bash
python train.py \
    --experiment_name=climate_run_1 \
    --wandb_project=climate_pinn \
    --hidden_dim=64 \
    --initial_re=100.0 \
    --nb_years=10 \
    --train_val_split=0.8 \
    --batch_size=128 \
    --epochs=100 \
    --learning_rate=1e-3 \
    --physics_weight=0.5 \
    --data_weight=1.0
```

For distributed training on a SLURM cluster, use the provided script:

```bash
./run_training.sh
```

## Visualization

To generate visualizations of model predictions:

```bash
python visualisation.py \
    --checkpoint_path=checkpoints/best_climate_run_1.pt \
    --data_dir=./data/era_5_data \
    --num_frames=24
```

This will create animations of:
- Predicted vs. true temperature fields
- Predicted vs. true wind fields
- Physics residuals

## Configuration

Key hyperparameters and their descriptions:

- `hidden_dim`: Dimension of hidden layers (default: 64)
- `initial_re`: Initial Reynolds number (default: 100.0)
- `physics_weight`: Weight of physics loss terms (default: 0.5)
- `data_weight`: Weight of data loss terms (default: 1.0)
- `batch_size`: Training batch size (default: 128)
- `learning_rate`: Initial learning rate (default: 1e-3)

## Results Tracking

The training process is tracked using Weights & Biases, logging:
- Training and validation losses
- Physics constraint residuals
- Reynolds number evolution
- Prediction visualizations
- Model checkpoints


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview) data provided by ECMWF
- Navier-Stokes equations for PINNs: [Physics-informed neural networks for high-resolution weather reconstruction from sparse weather stations](https://open-research-europe.ec.europa.eu/articles/4-99) 
