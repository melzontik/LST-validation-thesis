import os
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__)) + '//'

# Read the data from the Excel file
df_filtered = pd.read_excel('Celsius_Robust_outlier_remover_cross_sensors_normalization_inside_every_station_1100_vs_01.xlsx')

# List of station names
station_names = ['USA_BOND', 'USA_DRAK', 'USA_FPEK', 'USA_GCMK', 'USA_SFSD', 'USA_SGP', 'USA_TBLN']
normalized_data_frames = []

for station in station_names:
    # Create a new DataFrame for each station
    station_df = df_filtered.loc[df_filtered['Station'] == station].copy()

    
    lst = station_df['Landsat8_f_LSTmean']
    normalized_lst = (lst - lst.min()) / (lst.max() - lst.min())
    
    lst1 = station_df['Modis_f_LSTmean']
    normalized_lst1 = (lst1 - lst1.min()) / (lst1.max() - lst1.min())
    
    gbov = station_df['Landsat8_GBOV_LST']
    normalized_gbov = (gbov - gbov.min()) / (gbov.max() - gbov.min())
    
    gbov1 = station_df['Modis_GBOV_LST']
    normalized_gbov1 = (gbov1 - gbov1.min()) / (gbov1.max() - gbov1.min())
    
    delta_lst = station_df['Landsat8_Delta_LST']
    normalized_delta_lst = (delta_lst - delta_lst.min()) / (delta_lst.max() - delta_lst.min())
    
    delta_lst1 = station_df['Modis_Delta_LST']
    normalized_delta_lst1 = (delta_lst1 - delta_lst1.min()) / (delta_lst1.max() - delta_lst1.min())

    tcwv = station_df['Landsat8_tcwv']
    normalized_tcwv = (tcwv - tcwv.min()) / (tcwv.max() - tcwv.min())
    
    tcwv1 = station_df['Modis_tcwv']
    normalized_tcwv1 = (tcwv1 - tcwv1.min()) / (tcwv1.max() - tcwv1.min())

    vza = station_df['Landsat8_r_VZA_mean']
    normalized_vza = (vza - vza.min()) / (vza.max() - vza.min())
    
    vaa = station_df['Landsat8_u_VAA_mean']
    normalized_vaa = (vaa - vaa.min()) / (vaa.max() - vaa.min())
    
    sza = station_df['Landsat8_l_SZA_mean']
    normalized_sza = (sza - sza.min()) / (sza.max() - sza.min())
    
    saa = station_df['Landsat8_o_SAA_mean']
    normalized_saa = (saa - saa.min()) / (saa.max() - saa.min())
    
    va = station_df['Modis_j_VA']
    normalized_va = (va - va.min()) / (va.max() - va.min())

    # Add normalized columns to the station DataFrame
    station_df['Landsat8_Norm_LST'] = normalized_lst
    station_df['Modis_Norm_LST'] = normalized_lst1
    station_df['Landsat8_Norm_GBOV'] = normalized_gbov
    station_df['Modis_Norm_GBOV'] = normalized_gbov1        
    station_df['Landsat8_Norm_Delta_LST'] = normalized_delta_lst
    station_df['Modis_Norm_Delta_LST'] = normalized_delta_lst1
    station_df['Landsat8_Norm_TCWV'] = normalized_tcwv
    station_df['Modis_Norm_TCWV'] = normalized_tcwv1
    station_df['Landsat8_Norm_VZA'] = normalized_vza
    station_df['Landsat8_Norm_VAA'] = normalized_vaa
    station_df['Landsat8_Norm_SZA'] = normalized_sza
    station_df['Landsat8_Norm_SAA'] = normalized_saa
    station_df['Modis_Norm_VA'] = normalized_va

    # Append the DataFrame for this station to the list
    normalized_data_frames.append(station_df)

# Concatenate all DataFrames into a single DataFrame
df_filtered = pd.concat(normalized_data_frames)

# Export the result as an Excel file
df_filtered.to_excel('celsius_normalized_inside_station_1100_vs_01.xlsx', index=False)
