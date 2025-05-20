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
param_dir = f"/nas/cee-hydro/laketemp_bias/params/lstm_param_cloud"
# data path
air_temp_path = "/nas/cee-hydro/laketemp_bias/era5land/air_temp.csv"
wind_path = "/nas/cee-hydro/laketemp_bias/era5land/wind.csv"
srad_path = "/nas/cee-hydro/laketemp_bias/era5land/srad.csv"
water_temp_path = "/nas/cee-hydro/laketemp_bias/era5land/water_temp_cloud.csv"
# ensemble
ensemble_num = 10
# use cpu or gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# train test period
train_period = (pd.to_datetime("2003-01-01", format="%Y-%m-%d"), pd.to_datetime("2017-12-31", format="%Y-%m-%d"))
val_period = (pd.to_datetime("2018-01-01", format="%Y-%m-%d"), pd.to_datetime("2023-12-31", format="%Y-%m-%d"))

# test_period = (pd.to_datetime("2020-01-01", format="%Y-%m-%d"), pd.to_datetime("2023-12-31", format="%Y-%m-%d"))
# hyperparameters
n_epochs = 30 # Number of training epochs
learning_rates = [1e-1, 1e-2, 1e-3] # Dynamic learning rate
hidden_size = 10 # Number of LSTM cells
dropout_rate = 0.0 # Dropout rate of the final fully connected Layer [0.0, 1.0]
sequence_length = 365 # Length of the meteorological record provided to the network
batch_size = 256

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

def train_single(ensemble_id,
                 total_df,
                ):
    ##############
    # Data set up#
    ##############
    # Training data
    ds_train = laketemp(lake_id, 
                        total_df,
                        seq_length=sequence_length, 
                        period="train", 
                        dates=train_period)
    tr_loader = DataLoader(ds_train, 
                           batch_size=batch_size, 
                           shuffle=True)

    # Validation data. We use the feature means/stds of the training period for normalization
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_val = laketemp(lake_id, 
                      total_df,
                      seq_length=sequence_length, 
                      period="eval", 
                      dates=val_period,
                      means=means, 
                      stds=stds)
    
    val_loader = DataLoader(ds_val, 
                            batch_size=batch_size, 
                            shuffle=False)

    # val_loader_cloud = DataLoader(ds_val_cloud, batch_size=2048, shuffle=False)

    # Test data. We use the feature means/stds of the training period for normalization
    # ds_test = laketemp(lake_id, 
    #                    total_df,
    #                    seq_length=sequence_length, 
    #                    period="eval", dates=test_period,
    #                    means=means, stds=stds)
    # test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    #########################
    # Model, Optimizer, Loss
    #########################

    # based on the mean and stds, calculate the normalized value for 0 C
    normalized_zero = (0 - means["tw"])/stds["tw"]
    # Here we create our model, feel free 
    model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate, normalized_zero = normalized_zero).to(DEVICE)
    loss_func = nn.MSELoss()
    
    for i in range(n_epochs):
        if i < 10:
            learning_rate = learning_rates[0]
        elif 10 < i <= 20:
            learning_rate = learning_rates[1]
        elif 20 < i <= 30:
            learning_rate = learning_rates[2]
        else:
            learning_rate = learning_rates[-1]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_epoch(model, optimizer, tr_loader, loss_func, i+1)
        obs, preds = eval_model(model, val_loader)
        preds = ds_val.local_rescale(preds.cpu().numpy(), variable='output')
        rmse = calc_rmse(obs.numpy(), preds)
        try:
            nse = calc_nse(obs.numpy(), preds)
            print(f"Validation RMSE: {rmse:.2f}, NSE: {nse:.2f}")
        except:
            nse = None
            print(f"Validation RMSE: {rmse:.2f}, NSE: None")
    # save the model parameters
    torch.save(model.state_dict(), f"{param_dir}/{lake_id}_{ensemble_id}.pt")

if __name__ == "__main__":
    for ensemble_id in range(ensemble_num):
        # set manual seed
        torch.manual_seed(ensemble_id)
        train_single(ensemble_id, total_df)
        print(f"Ensemble {ensemble_id} done")