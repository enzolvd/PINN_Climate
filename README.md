# Climate-PINN: Physics-Informed Neural Networks for Climate Modeling

This repository implements a Physics-Informed Neural Network (PINN) approach to climate modeling, combining deep learning with physical constraints from fluid dynamics. The model is trained on ERA5 climate reanalysis data and predicts near-surface temperature and wind patterns while respecting fundamental physics principles derived from the Navier-Stokes equations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Physics-informed neural network architecture incorporating Navier-Stokes equations as physics constraints
- Training on ERA5 climate reanalysis data with multiple variables:
  - Geopotential at 500 hPa
  - Temperature at 2m and 850 hPa
  - U and V wind components at 10m
  - Land-sea mask, orography, and soil type as geographical constraints
- Visualization tools for model predictions including animated temperature and wind field maps
- Automatic checkpointing and experiment tracking with Weights & Biases
- Support for distributed training on GPU clusters using SLURM
- Experiment queue system for batch processing multiple model configurations

## Project Structure

```
climate-pinn/
├── data/                     # Data directory (not included in repo)
│   └── era_5_data/           # ERA5 climate data
│       ├── constants/        # Constant fields (orography, land-sea mask, soil type)
│       ├── geopotential_500/ # Geopotential at 500 hPa
│       ├── 2m_temperature/   # Temperature at 2m
│       ├── temperature_850/  # Temperature at 850 hPa
│       ├── 10m_u_component_of_wind/
│       └── 10m_v_component_of_wind/
├── models/                   # Model definitions
│   ├── model_0.py            # Original model
│   ├── model_0_Re.py         # Model with clipped gradient and momentum on Reynolds number
│   ├── model_1.py            # Model with modified dropout placement
│   ├── model_2.py            # Model with well-placed dropout and Reynolds number adjustments
│   ├── model_3.py            # Model with neural network for Reynolds number estimation
│   └── models.txt            # Description of model variants
├── experiment_runner/        # Experiment queue management: for cluster working with SLURM and crontab
├── dataset.py                # ERA5 dataset loading and preprocessing
├── train.py                  # Training script
├── visualisation.py          # Visualization generation script
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```

## Model Architecture

The Climate-PINN consists of several key components:

- **MeteoEncoder**: Processes meteorological input variables (geopotential and temperature)
- **MaskEncoder**: Handles geographical masks (orography, land-sea mask, soil type)
- **CoordProcessor**: Processes spatial (lat/lon) and temporal coordinates
- **Physics Constraints**: Incorporates Navier-Stokes equations with learnable Reynolds number

### Model Variants

The repository includes several model variants:

1. **model_0**: Original baseline model
2. **model_0_Re**: Enhanced with clipped gradient and momentum on the Reynolds number
3. **model_1**: Model with modified dropout placement
4. **model_2**: Model with correctly placed dropout and improved Reynolds number handling
5. **model_3**: Advanced model with a neural network for Reynolds number estimation

## Physics Constraints

The PINN is constrained by fluid dynamics principles from the Navier-Stokes equations:

1. **Continuity Equation**: Ensures conservation of mass
2. **Momentum Equations**: Govern the conservation of momentum in x and y directions
3. **Reynolds Number**: The model learns an appropriate Reynolds number to balance inertial and viscous forces

These physics constraints are incorporated into the loss function, alongside data-driven loss terms, ensuring the model's predictions respect physical laws.

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
pip install -r requirements.txt
```


## Data Preparation

The model expects ERA5 climate data in NetCDF format. You can download ERA5 data from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home) or more easily from [WeatherBench](https://github.com/pangeo-data/WeatherBench).

Data should be organized in the following structure:
```
data/
└── era_5_data/
    ├── constants/
    │   └── constants.nc      # Contains orography, land-sea mask, and soil type
    ├── geopotential_500/
    ├── 2m_temperature/
    ├── temperature_850/
    ├── 10m_u_component_of_wind/
    └── 10m_v_component_of_wind/
```

Each variable directory should contain yearly NetCDF files (e.g., `geopotential_500_1979_data.nc`).

## Training

### Single Experiment

To train a single model:

```bash
python train.py \
    --model=model_2 \
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
    --data_weight=1.0 \
    --data_dir=./data/era_5_data
```

### Using the Experiment Queue

For running multiple experiments sequentially:

1. Edit `experiment_runner/experiments.json` to define your experiments
2. Update the queue:
```bash
./experiment_runner/update_queue.sh
```

3. Start the queue runner:
```bash
./experiment_runner/run_queue.sh
```

### Distributed Training on SLURM

The repository includes configuration for training on SLURM clusters. The experiment queue system automatically submits jobs to available SLURM partitions.

## Visualization

To generate visualizations of model predictions:

```bash
python visualisation.py
```

This will create visualizations for all runs specified in the script, including:
- Temperature field predictions vs. ground truth
- Temperature prediction error maps
- Wind field predictions vs. ground truth
- Wind field prediction error maps

## Hyperparameter Configuration

Key hyperparameters and their descriptions:

- `model`: Model variant to use (e.g., model_0, model_2, model_3)
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

## Example Results

The PINN models demonstrate the ability to predict surface temperature and wind patterns while respecting physical constraints. Visualization tools allow for the creation of:

- Global temperature forecast animations
- Vector field animations of predicted wind patterns
- Difference maps highlighting prediction errors
- Physics residual maps showing where physical constraints are violated

## Acknowledgments

- [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview) data provided by ECMWF
- Physics-informed neural networks methodology adapted from [Raissi et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- Navier-Stokes equations for PINNs in climate modeling: [Physics-informed neural networks for high-resolution weather reconstruction from sparse weather stations](https://open-research-europe.ec.europa.eu/articles/4-99)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{louvard2025climatepinn,
  author = {Louvard, Enzo},
  title = {Climate-PINN: Physics-Informed Neural Networks for Climate Modeling},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/climate-pinn}}
}
```
