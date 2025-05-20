# Description
This repository contains code for evaluating the impact of cloud cover on lake surface water temperature using remote sensing and modeling techniques.

# Code Structure
All code is written in Jupyter Notebooks and organized within the notebooks/ directory.

1. notebooks/A...: Retrieves water temperature data from ERA5-Land and cloud cover data from MODIS Aqua/Terra.

2. notebooks/B-air2water and notebooks/B-lstm: Calibrates the air2water model and trains an LSTM model for each study lake.

3. notebooks/C...: Calculates annual summary statistics for both synthetic observations and model simulations.

4. notebooks/D...: Plots differences in annual summary statistics.

5. notebooks/E1-plot_example_lakes.ipynb: Plots lake temperature time series for individual lakes.