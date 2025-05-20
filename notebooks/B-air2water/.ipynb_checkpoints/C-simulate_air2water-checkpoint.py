import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from laketempmodels.models import air2water6p
from laketempmodels.tools.monte_carlo import monte_carlo
from laketempmodels.utils.metrics import calc_nse, calc_mse
import json


# ======================================= Load parameter and simulate =====================================
job = int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1
# hydrolake depth
hydrolake = pd.read_csv("data/cci_lakes_hydrolake_depth.csv", index_col = 0)
# list of cci lakes. Some lakes don't have ERA5-Land data, which are excluded
cci_lake_list = hydrolake.index.to_numpy().astype(np.int64)
# get lake id
lake_id = cci_lake_list[job]
# weather data
df_airtemp = pd.read_csv("/nas/cee-hydro/laketemp_bias/era5land/air_temp.csv", 
                         index_col=0, parse_dates=True, usecols=["Unnamed: 0", str(lake_id)])
# observation water temperature
df_tw = pd.read_csv("/nas/cee-hydro/laketemp_bias/era5land/water_temp.csv", 
                    index_col=0, parse_dates=True, usecols=["Unnamed: 0", str(lake_id)])
# find the file path of the parameters
param_path = f"data/a2w_param_full/{lake_id}.csv"
# load air2water model parameters
param_list = ["a1", "a2", "a3", "a4", "a5", "a6"]
param_dict = pd.read_csv(param_path, index_col=0).loc[lake_id, param_list].to_dict()
th = 4.0
tw_init = 0
tw_range = (0,30)
simulation_period = pd.date_range("2000-01-01", "2023-12-31")
# output simulation directory
output_path = f"data/a2w_full_sim/{lake_id}.csv"

def Load_data(lake_id,
              sim_period = simulation_period,
             ):
    # load air temperature
    ta = df_airtemp.loc[:, str(lake_id)]
    ta.index.name = "date"
    # load water temperature observation
    tw_obs = df_tw.loc[:, str(lake_id)]
    tw_obs.index.name = "date"
    
    # calculate the day of the year
    def cal_t_ty(dt):
        t = dt.dayofyear
        if dt.year%4 == 0:
            ty = 366
        else:
            ty = 365
        return t/ty
    
    # calculate daily mean temperature from tmax and tmin
    df = pd.concat([ta, tw_obs], axis = 1)
    df.columns = ["ta", "tw_obs"]
    # set negative temperature as 0 as the air2water model
    df["tw_obs"] = df["tw_obs"].clip(0,999)
    df["t_ty"] = df.index.map(cal_t_ty)

    sim_ta = df.loc[sim_period,"ta"].to_numpy().ravel()
    sim_tw_obs = df.loc[sim_period,"tw_obs"].to_numpy().ravel()
    sim_t_ty = df.loc[sim_period,"t_ty"].to_numpy().ravel()
    
    return sim_ta, sim_tw_obs, sim_t_ty


if __name__ == "__main__":
    model = air2water6p()
    model.set_params(param_dict)
    sim_ta, sim_tw_obs, sim_t_ty = Load_data(lake_id)
    sim_tw = model.simulate(sim_ta, sim_t_ty, th = th, tw_init = tw_init, tw_ice = 0.0)
    sim_tw_df = pd.DataFrame(sim_tw, index = simulation_period)
    sim_tw_df.columns = ["tw"]
    sim_tw_df.index.name = "date"
    sim_tw_df.to_csv(output_path)