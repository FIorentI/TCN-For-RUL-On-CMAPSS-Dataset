# RUL Prediction using TCN on CMAPSS Dataset

This project implements a Temporal Convolutional Network (TCN) to predict the Remaining Useful Life (RUL) of turbofan engines from the NASA CMAPSS dataset.

## Description

The goal is to predict the RUL of engines based on sensor readings over time. This is a regression problem where the model learns to map a sequence of sensor data to a sequence of RUL values. The model uses a TCN architecture, which is well-suited for sequence modeling tasks.

The project uses `wandb` for experiment tracking and visualization.

## Dependencies

To run this project, you need to install the following libraries. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The dependencies are:
- `torch`
- `numpy`
- `pandas`
- `wandb`
- `matplotlib`
- `seaborn`
- `torchinfo`
- `pytorch-tcn`
- `scikit-learn`

## Dataset

This project uses the [CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

You need to download the dataset and place it in a directory. The path is can be change using the `--dataset_dir` argument. The script is configured to use one of the four subsets (FD001, FD002, FD003, FD004) of the dataset.

## Usage

To train the model, run the `TCN_For_CMAPSS.py` script. You can customize the training process using command-line arguments.

```bash
python TCN_For_CMAPSS.py [OPTIONS]
```

### Command-line Arguments

- `--dataset_dir`: Path to the CMAPSS dataset directory. 
- `--set_number`: The dataset subset number (1-4). (Default: `4`)
- `--model_dir`: Directory to save the trained model. 
- `--model_name`: Name of the model file. (Default: `tcn_rul`)
- `--window_size`: Size of the input window. (Default: `30`)
- `--rul_clip`: Maximum RUL value to clip. (Default: `125`)
- `--batch_size`: Batch size for training. (Default: `8`)
- `--learning_rate`: Initial learning rate. (Default: `1e-2`)
- `--epochs`: Maximum number of training epochs. (Default: `1000`)
- `--num_filters`: Number of filters in TCN layers. (Default: `200`)
- `--kernel_size`: Kernel size for TCN convolutions. (Default: `3`)
- `--num_blocks`: Number of residual blocks in TCN. (Default: `4`)
- `--dropout`: Dropout rate. (Default: `0.3`)
- `--early_stop_patience`: Patience for early stopping. (Default: `40`)
- `--lr_decay_patience`: Patience for learning rate decay. (Default: `20`)
- `--lr_decay_factor`: Factor for learning rate decay. (Default: `0.5`)
- `--seed`: Random seed. (Default: `42`)
- `--verbose`: Verbose output.

Example:
```bash
python TCN_For_CMAPSS.py --set_number 1 --batch_size 16 --learning_rate 0.001
```

## Model Architecture

The model is a `Seq2SeqTCN` which uses the `pytorch-tcn` library. It consists of a stack of TCN residual blocks followed by a linear layer to produce the RUL prediction for each time step in the input sequence.

## Training

The training process includes:
- **Data Preprocessing**: Loading the data, calculating RUL, and scaling sensor values.
- **Data Loaders**: Creating PyTorch `DataLoader`s for training and validation sets.
- **Optimization**: Using the Adam optimizer and Mean Squared Error (MSE) loss (specifically, the root mean squared error is used for logging).
- **Learning Rate Scheduling**: `ReduceLROnPlateau` scheduler to decrease the learning rate when validation loss plateaus.
- **Early Stopping**: The training stops if the validation loss does not improve for a specified number of epochs (`early_stop_patience`).
- **Experiment Tracking**: `wandb` is used to log training/validation loss, and model configuration. The model with the best validation loss is saved.
