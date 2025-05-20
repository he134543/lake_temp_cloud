# Imports
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import os
from sklearn.model_selection import KFold
from itertools import product

os.chdir("/work/pi_kandread_umass_edu/lake_temp_bias/satbias_model/satlswt")

# ========================================= Training characteristics ==============================================
# read the job
job = int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1
# hydrolake depth
hydrolake = pd.read_csv("data/cci_lakes_hydrolake_depth.csv", index_col = 0)
# list of cci lakes. Some lakes don't have ERA5-Land data, which are excluded
cci_lake_list = hydrolake.index.to_numpy().astype(np.int64)
# cci lake id
lake_id = str(cci_lake_list[job])

# file directory to save parameter
# param_dir = f"/work/pi_kandread_umass_edu/lake_temp_bias/satbias_model/satlswt/params/lstm_param_{model}"
param_dir = f"/nas/cee-hydro/laketemp_bias/params/lstm_param_cloud"
sim_dir = f"/nas/cee-hydro/laketemp_bias/simulations/lstm_cloud_sim"

# data path
air_temp_path = "/nas/cee-hydro/laketemp_bias/era5land/air_temp.csv"
wind_path = "/nas/cee-hydro/laketemp_bias/era5land/wind.csv"
srad_path = "/nas/cee-hydro/laketemp_bias/era5land/srad.csv"
water_temp_path = f"/nas/cee-hydro/laketemp_bias/era5land/water_temp_cloud.csv"

# ensemble
ensemble_num = 10
# use cpu or gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# train test period
train_period = (pd.to_datetime("2003-01-01", format="%Y-%m-%d"), pd.to_datetime("2017-12-31", format="%Y-%m-%d"))
val_period = (pd.to_datetime("2018-01-01", format="%Y-%m-%d"), pd.to_datetime("2023-12-31", format="%Y-%m-%d"))
sim_period = (pd.to_datetime("2001-01-01", format="%Y-%m-%d"), # needs 365 days in advance
              pd.to_datetime("2023-12-31", format="%Y-%m-%d"))
# test_period = (pd.to_datetime("2020-01-01", format="%Y-%m-%d"), pd.to_datetime("2023-12-31", format="%Y-%m-%d"))

# hyperparameters
n_epochs = 100 # Number of training epochs
initial_lr = 1e-1  # or 1e-3 depending on how aggressive you want to start
decay_rate = 0.96  # decay per epoch
sequence_length = 365 # Length of the meteorological record provided to the network
batch_size = 256

# hyper parameters needs to be turned
hidden_size_list = [8, 16, 32] # Number of hidden layer units
dropout_rate = [0.0, 0.1, 0.2] # Dropout rate of the final fully connected Layer [0.0, 1.0]

# preload the weather dataframe
# ========================================== preload forcing ==========================================
def load_forcing(lake_id):
    # load weather data
    df_air = pd.read_csv(air_temp_path, index_col=0, parse_dates=True, usecols=["Unnamed: 0", str(lake_id)])
    df_wind = pd.read_csv(wind_path, index_col=0, parse_dates=True, usecols=["Unnamed: 0", str(lake_id)])
    df_srad = pd.read_csv(srad_path, index_col=0, parse_dates=True, usecols=["Unnamed: 0", str(lake_id)])
    # load weather data
    weather_df = pd.concat([df_air, df_wind, df_srad], axis = 1)
    weather_df.columns = ["ta", "wind", "srad"]
    return weather_df

# preload the water temperature
def load_water_temp(lake_id):
    # load water temperature
    twdf = pd.read_csv(water_temp_path,
                     index_col=0, parse_dates=True, usecols=["Unnamed: 0", str(lake_id)]).clip(0, 999)
    twdf.columns = ["tw"]
    return twdf.tw

# lake dataframe
total_df = load_forcing(lake_id)
total_df['tw'] = load_water_temp(lake_id)

@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


class laketemp(Dataset):
    def __init__(self, 
                 lake_id: str,
                 total_df: pd.DataFrame=None, # combination of weather and obs water temperature
                 seq_length: int=365,
                 period: str=None,
                 dates: List=None, 
                 means: pd.Series=None, 
                 stds: pd.Series=None):
        self.lake_id = lake_id
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds

        # load datafrmae
        self.total_df = total_df
        # load data into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output data from text files."""
        df = self.total_df
        
        if self.dates is not None:
            # If meteorological observations exist before start date
            # use these as well. Similiar to hydrological warmup period.
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            df = df[start_date:self.dates[1]]

        # if training period store means and stds
        if self.period == 'train':
            self.means = df.mean()
            self.stds = df.std()

        # extract input and output features from DataFrame
        x = np.array([df['ta'].values,
                      df['wind'].values,
                      df['srad'].values,
                      ]).T
        y = np.array([df['tw'].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self._local_normalization(x, variable='inputs')
        x, y = reshape_data(x, y, self.seq_length)

        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.lake_id}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)
            
            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)
            
            # normalize discharge
            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['ta'],
                              self.means['wind'],
                              self.means['srad']])
            stds = np.array([self.stds['ta'],
                             self.stds['wind'],
                             self.stds['srad']])
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["tw"]) /
                       self.stds["tw"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['ta'],
                              self.means['wind'],
                              self.means['srad']])
            stds = np.array([self.stds['ta'],
                             self.stds['wind'],
                             self.stds['srad']])
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["tw"] +
                       self.means["tw"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds
    
    
class Model(nn.Module):
    """Implementation of a single layer LSTM network"""
    
    def __init__(self, hidden_size: int, dropout_rate: float=0.0, normalized_zero: float = 0.0):
        """Initialize model
        
        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # create required layer
        self.lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size, 
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.normalized_zero = normalized_zero
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.
        
        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.
        
        :return: Tensor containing the network predictions
        """
        output, (h_n, c_n) = self.lstm(x)
        
        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1,:,:]))
        
        # clamp prediction of 0 C
        # pred = torch.clamp(pred, min = self.normalized_zero)
        
        return pred
    
    
def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    
    # pbar = tqdm.tqdm_notebook(loader)
    # pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in loader:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        # pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        
def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.
    
    :return: Two torch Tensors, containing the observations and 
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)
            
    return torch.cat(obs), torch.cat(preds)
        
def calc_rmse(obs, sim):
    """Calculate the mean squared error.
    Args:
        obs: Array of the observed values
        sim: Array of the simulated values
    Returns:
        The MSE value for the simulation, compared to the observation.
    """
    # Validation check on the input arrays  
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
    # drop nan and negative temperature from the observation
    nan_index = obs>=0  
    obs = obs[nan_index]
    sim = sim[nan_index]
    
    # Calculate the rmse value
    rmse_val = np.sqrt(np.mean((obs-sim)**2))

    return rmse_val

def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val

def run_kfold_cv(hparams, total_df, k=5, patience=10):
    rmse_scores = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_dates = total_df.index[(total_df.index >= train_period[0]) & (total_df.index <= train_period[1])]
    all_dates = sorted(all_dates)[sequence_length:]

    date_splits = list(kf.split(all_dates))

    for fold, (train_idx, val_idx) in enumerate(date_splits):
        train_dates = (all_dates[train_idx[0]], all_dates[train_idx[-1]])
        val_dates = (all_dates[val_idx[0]], all_dates[val_idx[-1]])

        ds_train = laketemp(lake_id, total_df, seq_length=sequence_length, period="train", dates=train_dates)
        tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        ds_val = laketemp(lake_id, total_df, seq_length=sequence_length, period="eval", dates=val_dates,
                          means=ds_train.get_means(), stds=ds_train.get_stds())
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        normalized_zero = (0 - ds_train.get_means()["tw"]) / ds_train.get_stds()["tw"]
        model = Model(hidden_size=hparams["hidden_size"],
                      dropout_rate=hparams["dropout_rate"],
                      normalized_zero=normalized_zero).to(DEVICE)
        loss_func = nn.MSELoss()

        best_rmse = float('inf')
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            lr = initial_lr * (decay_rate ** epoch)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train_epoch(model, optimizer, tr_loader, loss_func, epoch+1)

            obs, preds = eval_model(model, val_loader)
            preds = ds_val.local_rescale(preds.cpu().numpy(), variable='output')
            rmse = calc_rmse(obs.numpy(), preds)

            if rmse < best_rmse - 1e-3:
                best_rmse = rmse
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[Fold {fold+1}] Early stopping at epoch {epoch+1}")
                    break

        rmse_scores.append(best_rmse)

    return np.mean(rmse_scores)


def select_best_hyperparams(total_df):
    param_grid = list(product(hidden_size_list, dropout_rate))
    best_params = None
    best_score = float('inf')

    for hs, dr in param_grid:
        hparams = {"hidden_size": hs, "dropout_rate": dr}
        avg_rmse = run_kfold_cv(hparams, total_df, k=5)
        print(f"Params {hparams} => Avg. RMSE: {avg_rmse:.2f}")
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = hparams

    print(f"\nBest Hyperparameters: {best_params}, RMSE: {best_score:.2f}")
    return best_params


def train_and_simulate_best_params(ensemble_id, total_df, best_params, patience=10):
    ds_train = laketemp(lake_id, total_df, seq_length=sequence_length, period="train", dates=train_period)
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    ds_val = laketemp(lake_id, total_df, seq_length=sequence_length, period="eval", dates=val_period,
                      means=ds_train.get_means(), stds=ds_train.get_stds())
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    normalized_zero = (0 - ds_train.get_means()["tw"]) / ds_train.get_stds()["tw"]
    model = Model(hidden_size=best_params["hidden_size"],
                  dropout_rate=best_params["dropout_rate"],
                  normalized_zero=normalized_zero).to(DEVICE)

    loss_func = nn.MSELoss()
    best_rmse = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        lr = initial_lr * (decay_rate ** epoch)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_epoch(model, optimizer, tr_loader, loss_func, epoch+1)

        obs, preds = eval_model(model, val_loader)
        preds = ds_val.local_rescale(preds.cpu().numpy(), variable='output')
        rmse = calc_rmse(obs.numpy(), preds)

        print(f"[Epoch {epoch+1}] RMSE: {rmse:.2f}")

        if rmse < best_rmse - 1e-3:  # Small threshold to consider as "improvement"
            best_rmse = rmse
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model state and save
    model.load_state_dict(best_state)
    obs, preds = eval_model(model, val_loader)
    preds = ds_val.local_rescale(preds.cpu().numpy(), variable='output')
    final_rmse = calc_rmse(obs.numpy(), preds)
    final_nse = calc_nse(obs.numpy(), preds)
    param_str = f"hs{best_params['hidden_size']}_dr{int(best_params['dropout_rate']*100)}"
    torch.save(model.state_dict(), f"{param_dir}/{lake_id}_{ensemble_id}_{param_str}.pt")
    
    
    # --- Predict entire time series over sim_period ---
    ds_total = laketemp(lake_id, total_df, seq_length=sequence_length, period="eval",
                        dates=sim_period,
                        means=ds_train.get_means(), stds=ds_train.get_stds())
    total_loader = DataLoader(ds_total, batch_size=batch_size, shuffle=False)

    obs_total, preds_total = eval_model(model, total_loader)
    preds_total = ds_total.local_rescale(preds_total.cpu().numpy(), variable='output')
    obs_total = obs_total.numpy()

    # Get date index for reshaped outputs
    index = pd.date_range(start=sim_period[0],
                          end=sim_period[1] + pd.DateOffset(days=1), freq='D')

    result_df = pd.DataFrame({
        f'sim_{ensemble_id}': preds_total.flatten()
    }, index=index)

    return result_df  # <--- return the predictions


if __name__ == "__main__":
    best_params = select_best_hyperparams(total_df)
    all_preds = []
    for ensemble_id in range(ensemble_num):
        torch.manual_seed(ensemble_id)
        pred_df = train_and_simulate_best_params(ensemble_id, total_df, best_params)
        all_preds.append(pred_df)
        print(f"Ensemble {ensemble_id} done")

    # Merge all ensemble predictions
    ensemble_preds = pd.concat(all_preds, axis=1)

    # Save full dataframe
    out_csv = f"{sim_dir}/{lake_id}.csv"
    ensemble_preds.to_csv(out_csv)
    print(f"Saved ensemble predictions to: {out_csv}")