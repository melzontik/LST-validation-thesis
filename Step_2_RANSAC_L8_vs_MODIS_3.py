import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import pearsonr

# Define the path and output directory
path = os.path.dirname(os.path.abspath(__file__)) + '//'
output_path = path + 'NEW_CELSIUS_Landsat8_MODIS_Day_RANSAC_plot_buffer_1100_vs_01_wo' + '//'

# Read the data from your Excel file
df_cross_normalized = pd.read_excel('celsius_normalized_inside_station_1100_vs_01_l8_modis.xlsx')

station_names = ['USA_BOND', 'USA_DRAK', 'USA_FPEK', 'USA_GCMK', 'USA_SFSD', 'USA_SGP', 'USA_TBLN']

# Create lists of markers and info for each station
markers = ['o', '^', '*', 's', 'D', 'v', 'x']
Köppen_climate_classification = [
    'Temperate, humid, hot summer (Cfa)',
    'Cold arid (BWk)',
    'Arid, steppe, cold (BSk)',
    'Temperate, humid, hot summer (Cfa)',
    'Cold, humid (Dfa)',
    'Temperate, humid, hot summer (Cfa)',
    'Arid, steppe, cold (BSk)'
]
CCI_Land_Cover = [
    '11 (Cropland)',
    '120 (Shrubland)',
    '130 (Grassland)',
    '40 (Mosaic natural veg. / cropland)',
    '11 (Cropland)',
    '11 (Cropland)',
    '130 (Grassland)'
]

# Path filters for selected stations
path_filters = {
    'USA_DRAK': '035',
    'USA_SFSD': '030',
    'USA_FPEK': '026',
    'USA_SGP':  '034'
}

# Initialize lists to store data for all stations
station_names_export = []
ransac_coefs = []
ransac_intercepts = []
r_squared_inliers = []
pearson_coefficients = []
pearson_inliers = []

# Iterate through each station
for station_name, marker, climate, land_cover in zip(
    station_names, markers, Köppen_climate_classification, CCI_Land_Cover
):
    # Filter data for the current station
    station_data = df_cross_normalized[df_cross_normalized['Station'] == station_name]

    # Apply path-based filtering ONLY for selected stations
    if station_name in path_filters:
        path_code = path_filters[station_name]

        station_data = station_data[
            station_data['Landsat8_image_id']
            .astype(str)
            .str.contains(path_code, na=False)
        ]

    # Skip if no data after filtering
    if station_data.empty:
        print(f"Skipping {station_name}: no data after filtering")
        continue

    # Define X and y
    X = station_data['Landsat8_Norm_LST'].to_numpy()
    y = station_data['Modis_Norm_LST'].to_numpy()

    # RANSAC regression
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X.reshape(-1, 1), y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Stats
    lst_min  = station_data['Landsat8_f_LSTmean'].min()
    lst_max  = station_data['Landsat8_f_LSTmean'].max()
    lst_min1 = station_data['Modis_f_LSTmean'].min()
    lst_max1 = station_data['Modis_f_LSTmean'].max()
    delta_lst_buffer  = round(station_data['Landsat8_e_number_of_pixels'].mean(axis=0))
    delta_lst_buffer1 = round(station_data['Modis_e_number_of_pixels'].mean(axis=0))

    # Store results
    station_names_export.append(station_name)
    ransac_coefs.append(ransac.estimator_.coef_[0])
    ransac_intercepts.append(ransac.estimator_.intercept_)

    X_inliers = X[inlier_mask].reshape(-1, 1)
    y_inliers = y[inlier_mask]
    r_squared_inliers.append(ransac.score(X_inliers, y_inliers))

    pearson_coefficients.append(pearsonr(X, y)[0])
    pearson_inliers.append(pearsonr(X[inlier_mask], y[inlier_mask])[0])

    # Plotting
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.title(
        f"Landsat 8 LST vs MODIS Day LST @ {station_name} (2013-2022)\n"
        f"Köppen climate classification: {climate}\n"
        f"CCI Land Cover: {land_cover}",
        fontsize=22, fontweight='bold', y=1.02
    )

    plt.scatter(X[inlier_mask],  y[inlier_mask],  color="navy",       marker=marker, label="Inliers",  s=50)
    plt.scatter(X[outlier_mask], y[outlier_mask], color="yellowgreen", marker=marker, label="Outliers", s=50)

    plt.xlabel(
        f'Landsat 8 LST from {lst_min:.2f} to {lst_max:.2f} [C°] buffer {delta_lst_buffer} px',
        fontsize=22, fontweight='bold'
    )
    plt.ylabel(
        f'MODIS Day LST from {lst_min1:.2f} to {lst_max1:.2f} [C°] buffer {delta_lst_buffer1} px',
        fontsize=22, fontweight='bold'
    )

    x_line = np.array([[X.min()], [X.max()]])
    plt.plot(
        x_line, ransac.predict(x_line), 'r--',
        label="RANSAC: Y = {:.2f} X + {:.2f}".format(
            ransac.estimator_.coef_[0], ransac.estimator_.intercept_
        )
    )

    plt.plot(x_line, x_line, 'k-', label="Y = X")

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.legend(fontsize=22, frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Save
    station_output_path = os.path.join(output_path, station_name)
    os.makedirs(station_output_path, exist_ok=True)

    plt.savefig(os.path.join(station_output_path, f'lan8_mod_all_years_{station_name}.png'))
    plt.savefig(os.path.join(station_output_path, f'lan8_mod_all_years_{station_name}.pdf'))
    plt.close()

# Export results
data = {
    'station_name':       station_names_export,
    'RANSAC Coefficient': ransac_coefs,
    'RANSAC Intercept':   ransac_intercepts,
    'R-squared_inliers':  r_squared_inliers,
    'Pearson':            pearson_coefficients,
    'Pearson_inliers':    pearson_inliers,
}

df_export = pd.DataFrame(data)
excel_filename = output_path + 'celsius_cross_satellite_ransac_data.xlsx'
df_export.to_excel(excel_filename, index=False)

print("Processing completed.")