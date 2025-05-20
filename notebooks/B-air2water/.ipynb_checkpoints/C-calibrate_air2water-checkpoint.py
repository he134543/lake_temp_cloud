import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from air2waterpy import air2water
from air2waterpy.metrics import calc_mse, calc_nse
import os
os.chdir("/work/pi_kandread_umass_edu/lake_temp_bias/satbias_model/satlswt")

# read the job
job = int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1
# hydrolake depth
hydrolake = pd.read_csv("data/cci_lakes_hydrolake_depth.csv", 
                        index_col = 0)
# list of cci lakes. Some lakes don't have ERA5-Land data, which are excluded
cci_lake_list = hydrolake.index.to_numpy().astype(np.int64)
# cci lake id
cci_lake_id = cci_lake_list[job]

# weather data
# change here to replace with cloud gapped lake surface water temperature for calibration
df_tw = pd.read_csv("/nas/cee-hydro/laketemp_bias/era5land/water_temp.csv", index_col=0, 
                    parse_dates=True, usecols=["Unnamed: 0", str(cci_lake_id)])
# weather data
df_airtemp = pd.read_csv("/nas/cee-hydro/laketemp_bias/era5land/air_temp.csv", index_col=0, 
                         parse_dates=True, usecols=["Unnamed: 0", str(cci_lake_id)])
# calibration period -- > set the same as the train period with LSTM
calibration_period = pd.date_range("2000-01-01", "2014-12-31")
validation_period = pd.date_range("2020-01-01", "2023-12-31")
# set the ensemble id and this would equals to the random seed
ensemble_num = 100
# run parallel calibration
thread_count = 10
# Swarm size
size_swarm = 100
# iteration times
iter_times = 1000
# where to save the parameters
param_path = f"data/params/a2w_param_full/{cci_lake_id}.csv"

# ============================= Define Functions for loading data ==============================
def load_data(lake_id,
             ):
    # load air temperature
    ta = df_airtemp.loc[:, str(lake_id)]
    ta.index.name = "date"
    # load water temperature observation
    tw_obs = df_tw.loc[:, str(lake_id)]
    tw_obs.index.name = "date"
    # calculate daily mean temperature from tmax and tmin
    df = pd.concat([ta, tw_obs], axis = 1)
    df.columns = ["ta", "tw_obs"]
    # set negative temperature as 0 as the air2water model
    df["tw_obs"] = df["tw_obs"].clip(0,999)
    
    return df

def calibrate_parameters(lake_id, 
                         ensemble_id,
                         cal_period = pd.date_range("2000-01-01", "2014-12-31"),
                         val_period = pd.date_range("2020-01-01", "2023-12-31"),
                         sim_period = pd.date_range("2000-01-01", "2023-12-31"), # whole period
                         tw_init = 0, # set Jan-01 temperature as 0
                         tw_ice = 0,
                         th = 4.0,
                         swarm_size = 100,
                         n_cpus = 8,
                        ):
    # load data
    lake_df = load_data(lake_id)
    cal_ta, cal_tw_obs = lake_df.loc[cal_period].ta, lake_df.loc[cal_period].tw_obs
    val_ta, val_tw_obs = lake_df.loc[val_period].ta, lake_df.loc[val_period].tw_obs
    total_ta, total_tw_obs = lake_df.loc[sim_period].ta, lake_df.loc[sim_period].tw_obs
    
    # get hydrolake depth and area
    hydrolake_area = hydrolake.loc[int(lake_id), "Lake_area"]
    hydrolake_depth = hydrolake.loc[int(lake_id), "Depth_avg"]
    # based on the area, compute the potential range
    if hydrolake_area <= 1:
        depth_range = (0.1, hydrolake_depth + 2.38)
    elif 1< hydrolake_area <= 10:
        if hydrolake_area > (3.76 + 0.1):
            depth_range = (hydrolake_depth - 3.76, hydrolake_depth + 3.76)
        else:
            depth_range = (0.1, hydrolake_depth + 3.76)
    elif 10 < hydrolake_area <= 100:
        depth_range = (hydrolake_depth - 8.53, hydrolake_depth + 8.53)
    elif hydrolake_area > 100:
        depth_range = (hydrolake_depth - 15.56, hydrolake_depth + 15.56)
    else:
        raise ValueError(f"{lake_id} do not have valid lake area")
        return
    # based on training data, get the tw range
    tw_range = (np.nanmin(cal_tw_obs), np.nanmax(cal_tw_obs))
    
    # initialize a model
    model = air2water(version="8p")
    model.update_param_bnds(mean_depth_range = depth_range, tw_range = tw_range)
    
    # fit
    cost, joint_vars = model.pso_fit(cal_tw_obs.to_numpy(),
                                    cal_ta,
                                    cal_period,
                                    tw_init = tw_init, 
                                    tw_ice = 0, 
                                    swarm_size=swarm_size, 
                                    n_cpus = n_cpus,
                                    iteration_num = 500,
                                   )
    # get variables
    param_dict = dict(zip(model._param_list, joint_vars))
    # set parameters
    model.load_params(param_dict)

    # simulate water temperature during all period
    tw_sim = model.simulate(total_ta,
                            sim_period,
                            th = th,
                            tw_init = tw_init,
                            tw_ice = tw_ice, 
                            )
    cal_tw_sim = tw_sim.loc[cal_period]
    val_tw_sim = tw_sim.loc[val_period]
    
    output = pd.DataFrame(joint_vars, index = model._param_list).T
    output.index = [ensemble_id]
    # output.columns = model._param_list
    output["val_mse"] = calc_mse(val_tw_obs, val_tw_sim)
    output["cal_mse"] = calc_mse(cal_tw_obs, cal_tw_sim)
    return output

if __name__ == "__main__":
    output_df = pd.DataFrame([])
    sim_tw_df = pd.DataFrame([])
    for ensemble_id in range(ensemble_num):
        np.random.seed(ensemble_id)
        output, sim_tw = calibrate_parameters(cci_lake_id, ensemble_id, n_cpus = thread_count)
        output_df = pd.concat([output_df, output], axis = 0)
    # export parameters
    output.to_csv(param_path)