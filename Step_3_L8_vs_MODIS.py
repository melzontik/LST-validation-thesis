# GEE data extraction script
# https://code.earthengine.google.com/bf1360efd2bde25bf1b055724ffb6093

import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
import os

plt.close('all')

landsat_type = 'Landsat 8'
path = os.path.dirname(os.path.abspath(__file__))+'//'
input_path = path + 'Input//'
outut_path = path + 'nmad_Output//'+landsat_type+'//'
os.makedirs(outut_path, exist_ok=True)


# Read the CSV file
csv_file_name ='stats_over_Italy_'+landsat_type+'_modisTERRA_2023.csv'
df = pd.read_csv(input_path+ csv_file_name)



# Extract values from the 'prop' column without using ast.literal_eval
def extract_data_terra(x):
    try:
        # Extract the content within double curly braces
        data_terra_str = re.search(r'{([^}]*)}', str(x)).group(1)
        # Convert the string to a dictionary
        data_terra_dict = dict(item.split("=") for item in data_terra_str.split(", "))
        return data_terra_dict
    except AttributeError:
        return {}

df['data'] = df['prop'].apply(extract_data_terra)
df['coordinates'] = df['.geo'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})

# Convert date strings to DateTime objects
date_columns = ['date_landsat', 'date_terra']
for column in date_columns:
    df[column] = pd.to_datetime(df['data'].apply(lambda x: x.get(column)), format='ISO8601')

# Extract required columns and create a new DataFrame
result_df = pd.DataFrame({
    'landsat_ID': df['data'].apply(lambda x: x.get('landsat_ID')),
    'date_landsat': df['date_landsat'],
    'landsat_WRS_PATH': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_WRS_PATH'))),
    'landsat_WRS_ROW': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_WRS_ROW'))),
    'landsat_ROLL_ANGLE': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_ROLL_ANGLE'))),
    'landsat_NADIR_OFFNADIR': df['data'].apply(lambda x: x.get('landsat_NADIR_OFFNADIR')),
    'landsat_SUN_AZIMUTH': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_SUN_AZIMUTH'))),
    'landsat_SUN_ELEVATION': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_SUN_ELEVATION'))),
    'landsat_original_CLOUD_COVER': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_original_CLOUD_COVER'))),
    'landsat_cloud_cover_roi': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_cloud_cover_roi'))),
    'landsat_tot_pixels': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_tot_pixels'))),
    'landsat_umasked_pixels': pd.to_numeric(df['data'].apply(lambda x: x.get('landsat_umasked_pixels'))),
    'terra_ID': df['data'].apply(lambda x: x.get('terra_ID')),
    'date_terra': df['date_terra'],
    'terra_ROI_mean_time':  pd.to_numeric(df['data'].apply(lambda x: x.get('terra_mean_time'))),
    'terra_ROI_std_time':  pd.to_numeric(df['data'].apply(lambda x: x.get('terra_std_time'))),
    'terra_ROI_mean_angle':  pd.to_numeric(df['data'].apply(lambda x: x.get('terra_mean_angle'))),
    'terra_ROI_std_angle':  pd.to_numeric(df['data'].apply(lambda x: x.get('terra_std_angle'))),
    'difference_GSD':  pd.to_numeric(df['data'].apply(lambda x: x.get('difference_GSD'))),
    'mean_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('mean_difference'))),
    'median_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('median_difference'))),
    'mode_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('mode_difference'))),
    'std_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('std_difference'))),
    'nmad_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('nmad_difference'))),
    'max_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('max_difference'))),
    'min_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('min_difference'))),
    'unmasked_pixel_count_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('unmasked_pixel_count_difference'))),
    'tot_pixel_count_difference':  pd.to_numeric(df['data'].apply(lambda x: x.get('tot_pixel_count_difference')))
})

# to exclude no Data rows
mask = result_df['median_difference'] != -9999
result_df = result_df[mask]

mask = result_df['max_difference'] != -9999
result_df = result_df[mask]

mask = result_df['min_difference'] != -9999
result_df = result_df[mask]


# ------------------------------------------------------------------
# Human-readable labels for plot titles and axis names.
# The column names (used for file saving and loop filtering) are kept
# as-is; only the displayed text changes.
# {landsat_type} is a placeholder filled at runtime via .format().
# ------------------------------------------------------------------
col_labels = {
    'mean_difference':                'Mean LST Difference (MODIS - {landsat_type})',
    'median_difference':              'Median LST Difference (MODIS - {landsat_type})',
    'mode_difference':                'Mode LST Difference (MODIS - {landsat_type})',
    'std_difference':                 'Std Dev of LST Difference (MODIS - {landsat_type})',
    'nmad_difference':                'NMAD of LST Difference (MODIS - {landsat_type})',
    'max_difference':                 'Max LST Difference (MODIS - {landsat_type})',
    'min_difference':                 'Min LST Difference (MODIS - {landsat_type})',
    'unmasked_pixel_count_difference':'Co-valid pixel count for MODIS and {landsat_type}',
    'tot_pixel_count_difference':     'Total pixel count in ROI footprint ({landsat_type})',
    'difference_GSD':                 'GSD of LST difference image ({landsat_type})',
}


for col_name in result_df.columns:
    if 'difference' in col_name:

        # Drop NaNs just in case
        df_plot = result_df[['date_landsat', col_name]].dropna()
        # Sort by date (important for time series visualization)
        df_plot = df_plot.sort_values('date_landsat')

        # Get human-readable label; fall back to col_name if not in dict
        human_label = col_labels.get(col_name, col_name).format(landsat_type=landsat_type)

        # Pixel counts and GSD are dimensionless / metres — not °C
        y_unit = '[-]' if ('pixel_count' in col_name or 'GSD' in col_name) else '[°C]'

        # ------------------ TIME SERIES ------------------
        fig = plt.figure(figsize=(15, 8))
        ax = fig.gca()
        plt.title(
            f"{human_label} over Italy",
            fontsize=20, fontweight='bold'
        )
        plt.scatter(
            df_plot['date_landsat'],
            df_plot[col_name],
            s=40
        )
        plt.xlabel('Date (Landsat acquisition)', fontsize=18, fontweight='bold')
        plt.ylabel(f'{human_label} {y_unit}', fontsize=18, fontweight='bold')
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(outut_path + col_name + '_time_series.png', dpi=300)
        plt.close()

        # ------------------ HISTOGRAM ------------------
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca()
        plt.title(
            f"Distribution of \n{human_label}",
            fontsize=20, fontweight='bold'
        )
        plt.hist(
            df_plot[col_name],
            bins=30,
            edgecolor='black',
            linewidth=1.2
        )
        plt.xlabel(f'{human_label} {y_unit}', fontsize=18, fontweight='bold')
        plt.ylabel('Count [-]', fontsize=18, fontweight='bold')
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(outut_path + col_name + '_histogram.png', dpi=300)
        plt.close()


# Convert date columns to string BEFORE adding summary rows to keep column dtype clean
result_df['date_landsat'] = result_df['date_landsat'].astype(str)
result_df['date_terra'] = result_df['date_terra'].astype(str)

# numeric_only=True is mandatory — string columns (landsat_ID, NADIR_OFFNADIR) would crash mean/std/etc.
per_column_means   = result_df.mean(axis=0,   numeric_only=True)
per_column_stds    = result_df.std(axis=0,    numeric_only=True)
per_column_medians = result_df.median(axis=0, numeric_only=True)
per_column_nmads = result_df.filter(like='difference').apply(
    lambda col: col.dropna().subtract(col.dropna().median()).abs().median() * 1.4826,
    axis=0
)
per_column_maxs    = result_df.max(axis=0,    numeric_only=True)
per_column_mins    = result_df.min(axis=0,    numeric_only=True)

result_df.loc['column_mean']   = per_column_means
result_df.loc['column_std']    = per_column_stds
result_df.loc['column_median'] = per_column_medians
result_df.loc['column_max']    = per_column_maxs
result_df.loc['column_min']    = per_column_mins
result_df.loc['column_nmad'] = per_column_nmads

#print(result_df)
result_df.to_excel(outut_path + csv_file_name[:-4] + "_cleaned.xlsx", index=True)
