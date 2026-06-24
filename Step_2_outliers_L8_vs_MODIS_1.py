import os
import pandas as pd
from scipy.stats import median_abs_deviation
path = os.path.dirname(os.path.abspath(__file__)) + '//'

# List of station names
station_names = ['USA_BOND', 'USA_DRAK', 'USA_FPEK', 'USA_GCMK', 'USA_SFSD', 'USA_SGP', 'USA_TBLN']
# Define the list of columns to filter
columns_to_filter = ['Landsat8_f_LSTmean', 'Modis_f_LSTmean']

#'Landsat8_f_LSTmean', 'Landsat8_GBOV_LST', 'Landsat8_tcwv', 'Landsat8_r_VZA_mean', 'Modis_GBOV_LST', 'Modis_j_VA'
#'Landsat8_Delta_LST', 'Modis_Delta_LST'
#'Landsat8_h_LSTstd','Landsat8_j_roll', 'Modis_tcwv','Landsat8_n_SZA_std','Landsat8_t_VZA_std','Landsat8_w_VAA_std','Landsat8_q_SAA_std','Landsat8_k_SZA_calc','Landsat8_x_SZA_number_of_pixels','Landsat8_y_SAA_number_of_pixels','Landsat8_z_VZA_number_of_pixels',
#'Landsat8_zz_VAA_number_of_pixels','Landsat8_e_number_of_pixels', 'Modis_e_number_of_pixels','Landsat8_g_LSTmedian','Landsat8_m_SZA_median','Landsat8_v_VAA_median','Modis_g_LSTmedian',  
# Create an empty DataFrame to store the filtered data  'Landsat8_p_SAA_median','Landsat8_s_VZA_median','Landsat8_l_SZA_mean',  'Landsat8_o_SAA_mean', 'Landsat8_u_VAA_mean', 'Landsat8_cloud_cover', 
filtered_data = pd.DataFrame()

# Loop through station names to process data
for station_name in station_names:
    # Construct the folder path
    folder_path = os.path.join(path, 'CELSIUS_Cross_satellite_buffer_l8_modis_1100_vs_01_px_' + station_name)

    # Find the Excel file starting with 'cross_satellite_overall_all_years'
    excel_file = next((f for f in os.listdir(folder_path) if f.startswith('cross_satellite_overall_all_years')), None)

    if excel_file is not None:
        # Construct the full file path
        file_path = os.path.join(folder_path, excel_file)

        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # Add a 'Station' column to identify the station
        df['Station'] = station_name

        # Loop through the columns to filter
        for column in columns_to_filter:
            # Calculate median and median absolute deviation (MAD)
            median = df[column].median()
            nmad = median_abs_deviation(df[column], scale='normal')

            # Filter the data using the specified formula and append to the existing DataFrame
            df = df[(df[column] >= (median - 3 * nmad)) & (df[column] <= (median + 3 * nmad))]

        # Append the filtered data for this station to the main DataFrame
        filtered_data = pd.concat([filtered_data, df], ignore_index=True)

# Save the final filtered data to an Excel file
filtered_data.to_excel('Celsius_Robust_outlier_remover_cross_sensors_normalization_inside_every_station_1100_vs_01.xlsx', index=False)
