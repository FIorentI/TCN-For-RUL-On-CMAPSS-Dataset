import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary
from pytorch_tcn import TCN
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any

import argparse
from args import get_args


# -------------------------
# Data Preprocessing
# -------------------------

def load_data(file_path: str, id_col: List[str], cycle_col: List[str], setting_cols: List[str], sensor_cols: List[str]) -> pd.DataFrame:
    """Loads data from a single file."""
    data = pd.read_csv(file_path, sep=" ", header=None)
    data.drop([26, 27], axis=1, inplace=True)
    data.columns = id_col + cycle_col + setting_cols + sensor_cols
    return data

def add_rul_to_train_data(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the RUL column for training data."""
    max_cycle = data.groupby('id')['cycle'].transform('max')
    data['RUL'] = max_cycle - data['cycle']
    return data

def load_train_data(data_path: str, set_number: int, id_col: List[str], cycle_col: List[str], setting_cols: List[str], sensor_cols: List[str]) -> pd.DataFrame:
    """Loads and preprocesses a training dataset."""
    file_name = f'{data_path}/train_FD00{set_number}.txt'
    data = load_data(file_name, id_col, cycle_col, setting_cols, sensor_cols)
    data = add_rul_to_train_data(data)
    return data

# -------------------------
# Dataset Class
# -------------------------

class RULDataset(Dataset):
    """PyTorch Dataset for RUL prediction."""
    def __init__(self, data_list: List[np.ndarray], rul_list: List[np.ndarray]):
        self.data = data_list
        self.rul = rul_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.rul[idx], dtype=torch.float32)
        return X, y

# -------------------------
# TCN Model Definition
# -------------------------

class TimeDistributedBatchNorm1d(nn.Module):
    """Time-distributed BatchNorm1d."""
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x = x.view(B * L, C)
        x = self.bn(x)
        return x.view(B, L, C)

class Seq2SeqTCN(nn.Module):
    """ TCN model for RUL prediction."""
    def __init__(self, input_size: int, hidden_channels: int = 200, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=[hidden_channels] * 20,
            kernel_size=kernel_size,
            dilations=[1, 2, 4, 8, 16] * 4,
            dropout=dropout,
            use_norm='batch_norm',
            activation='relu',
            use_skip_connections=True,
            input_shape='NLC',
            causal=False
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            TimeDistributedBatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.linear(x)
        return x.squeeze(-1)

# -------------------------
# Training
# -------------------------

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, args):
    """Trains the model with early stopping."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = torch.sqrt(criterion(output, y_batch))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = torch.sqrt(criterion(output, y_batch))
                val_loss += loss.item() * len(X_batch)
            val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.model_dir + '/' + args.model_name+'.pth')
            print(f"New best model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{args.early_stop_patience}")

        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Final best validation loss: {best_val_loss:.4f}")

# -------------------------
# Main Execution
# -------------------------

def prepare_data(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Loads, preprocesses, and prepares data loaders."""
    id_col = ['id']
    cycle_col = ['cycle']
    setting_cols = ['setting1', 'setting2', 'setting3']
    sensor_cols = [f'sensor{i}' for i in range(1, 22)]

    train_df = load_train_data(args.dataset_dir, args.set_number, id_col, cycle_col, setting_cols, sensor_cols)
    train_df['RUL'] = train_df['RUL'].clip(upper=args.rul_clip)

    scaler = StandardScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])

    # Group by engine ID and convert to lists of sequences
    train_list = [group.drop(columns=['id', 'cycle', 'RUL']).values for _, group in train_df.groupby('id')]
    train_rul_list = [group['RUL'].tolist() for _, group in train_df.groupby('id')]

    # Pad sequences
    train_list_padded = pad_sequence([torch.tensor(seq) for seq in train_list], batch_first=True, padding_value=0)
    train_rul_list_padded = pad_sequence([torch.tensor(seq) for seq in train_rul_list], batch_first=True, padding_value=0)

    # Create dataset and split
    full_dataset = RULDataset(train_list_padded, train_rul_list_padded)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    """ RUL prediction pipeline"""

    args = get_args()

    wandb.init(project='pre_prod_rul', name=args.model_name, config=args)

    train_loader, val_loader = prepare_data(args)

    model = Seq2SeqTCN(
        input_size=24,  # Number of features
        kernel_size=args.kernel_size,
    )

    print("📊 Model Summary:")
    # Determine input size from a batch of data
    sample_batch, _ = next(iter(train_loader))
    summary(model, input_size=sample_batch.shape)

    train_model(model, train_loader, val_loader, args)
